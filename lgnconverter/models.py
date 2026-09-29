from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    ingesting = "ingesting"
    queued = "queued"
    processing = "processing"
    paused = "paused"
    cancelled = "cancelled"
    completed = "completed"
    failed = "failed"


class CreateJobRequest(BaseModel):
    source_url: HttpUrl
    authorization_confirmed: bool = Field(
        description="Confirms rights and platform authorization for this source."
    )
    source_language: str | None = Field(default=None, examples=["zh", "ko", "en"])
    target_language: str = "hi"
    clip_start_seconds: float = Field(default=0, ge=0)
    clip_duration_seconds: float = Field(default=120, gt=0, le=180)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranslatedSegment(BaseModel):
    start: float
    end: float
    source_text: str
    hindi_text: str


class Job(BaseModel):
    id: str
    status: JobStatus
    source_url: str
    source_language: str | None = None
    target_language: str = "hi"
    clip_start_seconds: float = 0
    clip_duration_seconds: float = 120
    detected_language: str | None = None
    duration_seconds: float | None = None
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    hindi_dialogue: list[TranslatedSegment] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkerResult(BaseModel):
    detected_language: str
    duration_seconds: float
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    hindi_dialogue: list[TranslatedSegment] = Field(default_factory=list)


class WorkerFailure(BaseModel):
    error: str
