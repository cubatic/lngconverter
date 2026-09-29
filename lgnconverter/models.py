from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class CreateJobRequest(BaseModel):
    source_url: HttpUrl
    authorization_confirmed: bool = Field(
        description="Confirms rights and platform authorization for this source."
    )
    source_language: str | None = Field(default=None, examples=["zh", "ko", "en"])
    target_language: str = "hi"


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Job(BaseModel):
    id: str
    status: JobStatus
    source_url: str
    source_language: str | None = None
    target_language: str = "hi"
    detected_language: str | None = None
    duration_seconds: float | None = None
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

