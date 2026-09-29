from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

from .config import Settings
from .models import CreateJobRequest, Job, JobStatus
from .pipeline import PipelineCancelled, extract_audio, transcribe
from .source import resolve_source
from .translation import translate_to_hindi


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._cancel_events: dict[str, Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lgn-pipeline")

    def create(self, request: CreateJobRequest) -> Job:
        job = Job(
            id=uuid4().hex,
            status=JobStatus.queued,
            source_url=str(request.source_url),
            source_language=request.source_language,
            target_language=request.target_language,
            clip_start_seconds=request.clip_start_seconds,
            clip_duration_seconds=request.clip_duration_seconds,
        )
        with self._lock:
            self._jobs[job.id] = job
            cancel_event = Event()
            self._cancel_events[job.id] = cancel_event
        self._executor.submit(self._run, job.id, cancel_event)
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def pause(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status in {JobStatus.queued, JobStatus.processing}:
                self._cancel_events[job_id].set()
                self._jobs[job_id] = job.model_copy(
                    update={"status": JobStatus.paused, "updated_at": datetime.now(UTC)}
                )
            return self._jobs[job_id].model_copy(deep=True)

    def resume(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status != JobStatus.paused:
                return job.model_copy(deep=True)
            cancel_event = Event()
            self._cancel_events[job_id] = cancel_event
            self._jobs[job_id] = job.model_copy(
                update={"status": JobStatus.queued, "updated_at": datetime.now(UTC), "error": None}
            )
            result = self._jobs[job_id].model_copy(deep=True)
        self._executor.submit(self._run, job_id, cancel_event)
        return result

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            self._cancel_events[job_id].set()
            self._jobs[job_id] = job.model_copy(
                update={"status": JobStatus.cancelled, "updated_at": datetime.now(UTC)}
            )
            return self._jobs[job_id].model_copy(deep=True)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            changes["updated_at"] = datetime.now(UTC)
            self._jobs[job_id] = job.model_copy(update=changes)

    def _run(self, job_id: str, cancel_event: Event) -> None:
        with self._lock:
            if self._cancel_events.get(job_id) is not cancel_event or cancel_event.is_set():
                return
        self._update(job_id, status=JobStatus.processing)
        job = self.get(job_id)
        assert job is not None
        audio_path = Path(self.settings.artifact_dir) / job_id / "source.wav"
        try:
            source = resolve_source(job.source_url, self.settings)
            duration = extract_audio(
                source,
                audio_path,
                job.clip_start_seconds,
                min(job.clip_duration_seconds, self.settings.max_source_seconds),
                self.settings.decode_timeout_seconds,
                cancel_event,
            )
            transcript, detected_language = transcribe(
                audio_path, self.settings, job.source_language, cancel_event
            )
            hindi_dialogue = []
            if self.settings.enable_translation:
                hindi_dialogue = translate_to_hindi(
                    transcript,
                    job.source_language or detected_language,
                    self.settings,
                    cancel_event,
                )
            self._update(
                job_id,
                status=JobStatus.completed,
                duration_seconds=duration,
                transcript=transcript,
                detected_language=detected_language,
                hindi_dialogue=hindi_dialogue,
            )
        # A background worker must persist unexpected stage failures on the job
        # instead of silently terminating the executor future.
        except PipelineCancelled:
            with self._lock:
                if self._cancel_events.get(job_id) is cancel_event:
                    current = self._jobs[job_id]
                    if current.status not in {JobStatus.paused, JobStatus.cancelled}:
                        self._jobs[job_id] = current.model_copy(update={"status": JobStatus.paused})
        except Exception as exc:  # noqa: BLE001
            self._update(job_id, status=JobStatus.failed, error=str(exc))
        finally:
            if not self.settings.keep_audio_artifacts:
                audio_path.unlink(missing_ok=True)
