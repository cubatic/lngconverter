import subprocess

import pytest

from lgnconverter.pipeline import extract_audio
from lgnconverter.source import ResolvedSource


def test_extract_audio_from_media_stream(tmp_path):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "decoded.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                str(source_path),
            ],
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("FFmpeg is not installed")

    duration = extract_audio(
        ResolvedSource(media_url=str(source_path)), output_path, start_seconds=0, duration_seconds=3
    )

    assert output_path.exists()
    assert duration == pytest.approx(1.0, abs=0.05)
