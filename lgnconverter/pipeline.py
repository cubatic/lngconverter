import json
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from threading import Event
from urllib.parse import urlparse

from .config import Settings
from .models import TranscriptSegment
from .source import ResolvedSource


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(PipelineError):
    pass


def _run_cancellable(command: list[str], timeout: float, cancel_event: Event | None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if cancel_event and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            raise PipelineCancelled("Processing paused or cancelled")
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout)
        time.sleep(0.2)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return stdout


def extract_audio(
    source: ResolvedSource,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    timeout_seconds: int = 900,
    cancel_event: Event | None = None,
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
            cancel_event,
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
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    try:
        _run_cancellable(command, timeout_seconds, cancel_event)
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
    cancel_event: Event | None,
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
        "--force-keyframes-at-cuts",
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
        _run_cancellable(command, timeout_seconds, cancel_event)
        _run_cancellable(
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
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            90,
            cancel_event,
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


@lru_cache(maxsize=3)
def _load_whisper_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("Install the inference dependencies: pip install -e '.[inference]'") from exc
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(
    audio_path: Path, settings: Settings, language: str | None, cancel_event: Event | None = None
):
    if cancel_event and cancel_event.is_set():
        raise PipelineCancelled("Processing paused or cancelled")
    model = _load_whisper_model(
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    if cancel_event and cancel_event.is_set():
        raise PipelineCancelled("Processing paused or cancelled")
    result = [
        TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    return result, info.language
