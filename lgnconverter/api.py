import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse

from .config import Settings, get_settings
from .jobs import JobStore
from .models import CreateJobRequest, Job, WorkerFailure, WorkerResult
from .source import SourceError, validate_source_url

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    store = JobStore(active_settings)
    app = FastAPI(title="LGN Converter", version="0.1.0")

    def require_client_key(x_lgn_api_key: str | None = Header(default=None)):
        if not active_settings.api_key:
            return
        if not x_lgn_api_key or not secrets.compare_digest(x_lgn_api_key, active_settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def require_worker_key(x_lgn_worker_key: str | None = Header(default=None)):
        if not active_settings.worker_api_key:
            raise HTTPException(status_code=503, detail="Worker API is not configured")
        if not x_lgn_worker_key or not secrets.compare_digest(
            x_lgn_worker_key, active_settings.worker_api_key
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing worker key")

    @app.get("/health")
    def health():
        return {"status": "ok", "youtube_source_enabled": active_settings.enable_youtube_source}

    @app.get("/", include_in_schema=False)
    def browser_prototype():
        return FileResponse(WEB_DIR / "index.html")

    @app.post(
        "/v1/jobs",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_client_key)],
    )
    def create_job(request: CreateJobRequest):
        if not request.authorization_confirmed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Source authorization must be confirmed",
            )
        if request.target_language != "hi":
            raise HTTPException(status_code=422, detail="Prototype target_language must be 'hi'")
        try:
            validate_source_url(str(request.source_url), active_settings)
        except SourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return store.create(request)

    @app.get(
        "/v1/jobs/{job_id}", response_model=Job, dependencies=[Depends(require_client_key)]
    )
    def get_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/jobs/{job_id}/pause", response_model=Job, dependencies=[Depends(require_client_key)]
    )
    def pause_job(job_id: str):
        job = store.pause(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/jobs/{job_id}/resume", response_model=Job, dependencies=[Depends(require_client_key)]
    )
    def resume_job(job_id: str):
        job = store.resume(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/jobs/{job_id}/cancel", response_model=Job, dependencies=[Depends(require_client_key)]
    )
    def cancel_job(job_id: str):
        job = store.cancel(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/worker/jobs/claim",
        response_model=Job | None,
        dependencies=[Depends(require_worker_key)],
    )
    def claim_worker_job():
        return store.claim()

    @app.get(
        "/v1/worker/jobs/{job_id}/audio",
        dependencies=[Depends(require_worker_key)],
        response_class=FileResponse,
    )
    def download_worker_audio(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        audio_path = store.audio_path(job_id)
        if not audio_path.is_file():
            raise HTTPException(status_code=409, detail="Audio is not ready")
        return FileResponse(audio_path, media_type="audio/wav", filename=f"{job_id}.wav")

    @app.get(
        "/v1/worker/jobs/{job_id}",
        response_model=Job,
        dependencies=[Depends(require_worker_key)],
    )
    def get_worker_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/worker/jobs/{job_id}/complete",
        response_model=Job,
        dependencies=[Depends(require_worker_key)],
    )
    def complete_worker_job(job_id: str, result: WorkerResult):
        job = store.complete_remote(job_id, result)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/v1/worker/jobs/{job_id}/fail",
        response_model=Job,
        dependencies=[Depends(require_worker_key)],
    )
    def fail_worker_job(job_id: str, failure: WorkerFailure):
        job = store.fail_remote(job_id, failure.error)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    return app


app = create_app()
