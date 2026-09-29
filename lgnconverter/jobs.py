import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

from .config import Settings
from .models import CreateJobRequest, Job, JobStatus, WorkerResult
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
        self._database_path = settings.artifact_dir / "jobs.sqlite3"
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._load_jobs()

    def _connect(self):
        return sqlite3.connect(self._database_path, timeout=10)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def _load_jobs(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM jobs").fetchall()
        for (payload,) in rows:
            job = Job.model_validate_json(payload)
            if job.status in {
                JobStatus.ingest_queued,
                JobStatus.ingesting,
                JobStatus.queued,
                JobStatus.processing,
            }:
                job = job.model_copy(update={"status": JobStatus.paused})
            self._jobs[job.id] = job
            self._cancel_events[job.id] = Event()
            self._persist(job)

    def _persist(self, job: Job) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs(id, payload) VALUES(?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (job.id, job.model_dump_json()),
            )

    def create(self, request: CreateJobRequest) -> Job:
        job = Job(
            id=uuid4().hex,
            status=(
                JobStatus.ingesting
                if self.settings.execution_mode == "remote"
                else JobStatus.queued
            ),
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
            self._persist(job)
        if self.settings.execution_mode == "local":
            self._executor.submit(self._run, job.id, cancel_event)
        else:
            self._update(job.id, status=JobStatus.ingest_queued)
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
            if job.status in {
                JobStatus.ingest_queued,
                JobStatus.ingesting,
                JobStatus.queued,
                JobStatus.processing,
            }:
                self._cancel_events[job_id].set()
                self._jobs[job_id] = job.model_copy(
                    update={"status": JobStatus.paused, "updated_at": datetime.now(UTC)}
                )
                self._persist(self._jobs[job_id])
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
            next_status = (
                JobStatus.queued
                if self.audio_path(job_id).exists()
                else JobStatus.ingest_queued
            )
            self._jobs[job_id] = job.model_copy(
                update={"status": next_status, "updated_at": datetime.now(UTC), "error": None}
            )
            self._persist(self._jobs[job_id])
            result = self._jobs[job_id].model_copy(deep=True)
        if self.settings.execution_mode == "local":
            self._executor.submit(self._run, job_id, cancel_event)
        return result

    def claim_ingest(self) -> Job | None:
        with self._lock:
            waiting = next(
                (job for job in self._jobs.values() if job.status == JobStatus.ingest_queued),
                None,
            )
            if not waiting:
                return None
            claimed = waiting.model_copy(
                update={"status": JobStatus.ingesting, "updated_at": datetime.now(UTC)}
            )
            self._jobs[claimed.id] = claimed
            self._persist(claimed)
            return claimed.model_copy(deep=True)

    def complete_ingest(self, job_id: str, audio: bytes, duration: float) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status == JobStatus.cancelled:
                return job.model_copy(deep=True)
            path = self.audio_path(job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
            queued = job.model_copy(
                update={
                    "status": JobStatus.queued,
                    "duration_seconds": duration,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = queued
            self._persist(queued)
            return queued.model_copy(deep=True)

    def audio_path(self, job_id: str) -> Path:
        return Path(self.settings.artifact_dir) / job_id / "source.wav"

    def dubbed_audio_path(self, job_id: str) -> Path:
        return Path(self.settings.artifact_dir) / job_id / "dubbed.m4a"

    def save_dubbed_audio(self, job_id: str, audio: bytes) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            path = self.dubbed_audio_path(job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
            updated = job.model_copy(
                update={"has_dubbed_audio": True, "updated_at": datetime.now(UTC)}
            )
            self._jobs[job_id] = updated
            self._persist(updated)
            return updated.model_copy(deep=True)

    def _prepare_remote(self, job_id: str, cancel_event: Event) -> None:
        job = self.get(job_id)
        if not job:
            return
        try:
            source = resolve_source(job.source_url, self.settings)
            duration = extract_audio(
                source,
                self.audio_path(job_id),
                job.clip_start_seconds,
                min(job.clip_duration_seconds, self.settings.max_source_seconds),
                self.settings.decode_timeout_seconds,
                cancel_event,
            )
            with self._lock:
                if self._cancel_events.get(job_id) is cancel_event and not cancel_event.is_set():
                    current = self._jobs[job_id]
                    queued = current.model_copy(
                        update={
                            "status": JobStatus.queued,
                            "duration_seconds": duration,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    self._jobs[job_id] = queued
                    self._persist(queued)
        except PipelineCancelled:
            return
        except Exception as exc:  # noqa: BLE001
            self._update(job_id, status=JobStatus.failed, error=str(exc))

    def claim(self) -> Job | None:
        with self._lock:
            queued = next(
                (job for job in self._jobs.values() if job.status == JobStatus.queued), None
            )
            if not queued:
                return None
            claimed = queued.model_copy(
                update={"status": JobStatus.processing, "updated_at": datetime.now(UTC)}
            )
            self._jobs[claimed.id] = claimed
            self._persist(claimed)
            return claimed.model_copy(deep=True)

    def complete_remote(self, job_id: str, result: WorkerResult) -> Job | None:
        dubbed_path = self.dubbed_audio_path(job_id)
        if not dubbed_path.is_file() and not result.hindi_dialogue:
            source_path = self.audio_path(job_id)
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-vn",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(dubbed_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or "Could not preserve source audio")
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status == JobStatus.cancelled:
                return job.model_copy(deep=True)
            completed = job.model_copy(
                update={
                    "status": JobStatus.completed,
                    "detected_language": result.detected_language,
                    "duration_seconds": result.duration_seconds,
                    "transcript": result.transcript,
                    "hindi_dialogue": result.hindi_dialogue,
                    "has_dubbed_audio": dubbed_path.is_file(),
                    "stage_timings": result.stage_timings,
                    "updated_at": datetime.now(UTC),
                    "error": None,
                }
            )
            self._jobs[job_id] = completed
            self._persist(completed)
            self.audio_path(job_id).unlink(missing_ok=True)
            return completed.model_copy(deep=True)

    def fail_remote(self, job_id: str, error: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            failed = job.model_copy(
                update={
                    "status": JobStatus.failed,
                    "error": error,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = failed
            self._persist(failed)
            return failed.model_copy(deep=True)

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            self._cancel_events[job_id].set()
            self._jobs[job_id] = job.model_copy(
                update={"status": JobStatus.cancelled, "updated_at": datetime.now(UTC)}
            )
            self._persist(self._jobs[job_id])
            self.audio_path(job_id).unlink(missing_ok=True)
            self.dubbed_audio_path(job_id).unlink(missing_ok=True)
            return self._jobs[job_id].model_copy(deep=True)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            changes["updated_at"] = datetime.now(UTC)
            self._jobs[job_id] = job.model_copy(update=changes)
            self._persist(self._jobs[job_id])

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
                        self._persist(self._jobs[job_id])
        except Exception as exc:  # noqa: BLE001
            self._update(job_id, status=JobStatus.failed, error=str(exc))
        finally:
            if not self.settings.keep_audio_artifacts:
                audio_path.unlink(missing_ok=True)
