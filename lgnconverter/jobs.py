from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .config import Settings
from .models import CreateJobRequest, Job, JobStatus
from .pipeline import extract_audio, transcribe
from .source import resolve_source


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lgn-pipeline")

    def create(self, request: CreateJobRequest) -> Job:
        job = Job(
            id=uuid4().hex,
            status=JobStatus.queued,
            source_url=str(request.source_url),
            source_language=request.source_language,
            target_language=request.target_language,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            changes["updated_at"] = datetime.now(UTC)
            self._jobs[job_id] = job.model_copy(update=changes)

    def _run(self, job_id: str) -> None:
        self._update(job_id, status=JobStatus.processing)
        job = self.get(job_id)
        assert job is not None
        audio_path = Path(self.settings.artifact_dir) / job_id / "source.wav"
        try:
            source = resolve_source(job.source_url, self.settings)
            duration = extract_audio(source, audio_path, self.settings.max_source_seconds)
            transcript, detected_language = transcribe(
                audio_path, self.settings, job.source_language
            )
            self._update(
                job_id,
                status=JobStatus.completed,
                duration_seconds=duration,
                transcript=transcript,
                detected_language=detected_language,
            )
        # A background worker must persist unexpected stage failures on the job
        # instead of silently terminating the executor future.
        except Exception as exc:  # noqa: BLE001
            self._update(job_id, status=JobStatus.failed, error=str(exc))
        finally:
            if not self.settings.keep_audio_artifacts:
                audio_path.unlink(missing_ok=True)
