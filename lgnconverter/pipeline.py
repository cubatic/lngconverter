import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .config import Settings
from .models import TranscriptSegment
from .source import ResolvedSource


class PipelineError(RuntimeError):
    pass


def extract_audio(
    source: ResolvedSource,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    timeout_seconds: int = 900,
) -> float | None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise PipelineError("FFmpeg and ffprobe must be installed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source.kind == "youtube" and source.page_url:
        _extract_youtube_range(
            source.page_url,
            output_path,
            start_seconds,
            duration_seconds,
            timeout_seconds,
        )
        return _probe_and_validate_duration(
            source, output_path, start_seconds, duration_seconds
        )

    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if urlparse(source.media_url).scheme in {"http", "https"}:
        command.extend(
            [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_on_network_error",
                "1",
                "-reconnect_on_http_error",
                "4xx,5xx",
                "-reconnect_delay_max",
                "10",
            ]
        )
    if source.http_headers:
        header_blob = "".join(f"{key}: {value}\r\n" for key, value in source.http_headers.items())
        command.extend(["-headers", header_blob])
    if start_seconds:
        command.extend(["-ss", str(start_seconds)])
    command.extend(
        [
            "-i",
            source.media_url,
            "-t",
            str(duration_seconds),
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
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        raise PipelineError(exc.stderr.strip() or "FFmpeg could not decode the source") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Source decoding timed out") from exc

    return _probe_and_validate_duration(source, output_path, start_seconds, duration_seconds)


def _extract_youtube_range(
    page_url: str,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    timeout_seconds: int,
) -> None:
    end_seconds = start_seconds + duration_seconds
    template = output_path.parent / "youtube-section.%(ext)s"
    section_wav = output_path.parent / "youtube-section.wav"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--force-overwrites",
        "--download-sections",
        f"*{start_seconds}-{end_seconds}",
        "--format",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--output",
        str(template),
        page_url,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(section_wav),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.CalledProcessError as exc:
        raise PipelineError(exc.stderr.strip() or "YouTube range extraction failed") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("YouTube range extraction timed out") from exc
    finally:
        section_wav.unlink(missing_ok=True)


def _probe_and_validate_duration(
    source: ResolvedSource,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> float:
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
    decoded_duration = float(json.loads(probe.stdout)["format"]["duration"])
    if source.duration is not None:
        expected_duration = min(duration_seconds, max(0, source.duration - start_seconds))
        if decoded_duration < expected_duration - 1:
            raise PipelineError(
                f"Remote stream ended early: expected {expected_duration:.1f}s, "
                f"received {decoded_duration:.1f}s. Retry the job."
            )
    return decoded_duration


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
