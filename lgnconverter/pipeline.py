import json
import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .models import TranscriptSegment
from .source import ResolvedSource


class PipelineError(RuntimeError):
    pass


def extract_audio(source: ResolvedSource, output_path: Path, max_seconds: int) -> float | None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise PipelineError("FFmpeg and ffprobe must be installed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if source.http_headers:
        header_blob = "".join(f"{key}: {value}\r\n" for key, value in source.http_headers.items())
        command.extend(["-headers", header_blob])
    command.extend(
        [
            "-i",
            source.media_url,
            "-t",
            str(max_seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=max_seconds + 90)
    except subprocess.CalledProcessError as exc:
        raise PipelineError(exc.stderr.strip() or "FFmpeg could not decode the source") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Source decoding timed out") from exc

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(probe.stdout)["format"]["duration"])


def transcribe(audio_path: Path, settings: Settings, language: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("Install the inference dependencies: pip install -e '.[inference]'") from exc

    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    result = [
        TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    return result, info.language

