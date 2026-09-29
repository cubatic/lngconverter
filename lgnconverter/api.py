from fastapi import FastAPI, HTTPException, status

from .config import Settings, get_settings
from .jobs import JobStore
from .models import CreateJobRequest, Job
from .source import SourceError, validate_source_url


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    store = JobStore(active_settings)
    app = FastAPI(title="LGN Converter", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "youtube_source_enabled": active_settings.enable_youtube_source}

    @app.post("/v1/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
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

    @app.get("/v1/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    return app


app = create_app()
