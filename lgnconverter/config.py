from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LGN_")

    enable_youtube_source: bool = False
    artifact_dir: Path = Path("artifacts")
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    max_source_seconds: int = 180
    decode_timeout_seconds: int = 900
    keep_audio_artifacts: bool = True
    enable_translation: bool = False
    translation_model: str = "facebook/nllb-200-distilled-600M"
    translation_device: str = "cpu"
    enable_tts: bool = False
    tts_model: str = "ai4bharat/indic-parler-tts"
    tts_device: str = "auto"
    tts_voice_description: str = (
        "Rani speaks Hindi in a youthful, light female voice. She sounds natural, cinematic "
        "and conversational, never like a news reader. "
        "The recording is very high quality, close-sounding, and has no background noise."
    )
    enable_source_separation: bool = False
    background_volume: float = 0.25
    dialogue_volume: float = 2.0
    enable_prosody_prompts: bool = False
    api_key: str = ""
    worker_api_key: str = ""
    ingest_api_key: str = ""
    execution_mode: str = "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
