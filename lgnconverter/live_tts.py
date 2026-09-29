import time
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .dubbing import _fit_duration, _run
from .models import TranslatedSegment
from .pipeline import PipelineError


@lru_cache(maxsize=1)
def _load_live_tts(model_name: str, device_name: str):
    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise PipelineError("Install the live TTS dependencies from the Colab notebook") from exc

    device = device_name
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer, device


def _synthesize_live(text: str, output_path: Path, settings: Settings) -> None:
    try:
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise PipelineError("Install soundfile and torch for live TTS") from exc

    model, tokenizer, device = _load_live_tts(settings.live_tts_model, settings.tts_device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0].float().cpu().numpy()
    sf.write(output_path, waveform, model.config.sampling_rate)


def _copy_source_audio(source_audio: Path, output_path: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_audio),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_path),
        ]
    )


def create_live_dubbed_audio(
    source_audio: Path,
    dialogue: list[TranslatedSegment],
    output_path: Path,
    settings: Settings,
) -> dict[str, float]:
    """Create a low-latency Hindi mix without neural source separation."""
    started = time.monotonic()
    if not dialogue:
        _copy_source_audio(source_audio, output_path)
        return {"separation": 0.0, "tts": 0.0, "mixing": round(time.monotonic() - started, 3)}

    work_dir = output_path.parent / "live-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = ["-i", str(source_audio)]
    filters = ["[0:a]volume=0.180[background]"]
    mix_labels = ["[background]"]

    for index, segment in enumerate(dialogue, start=1):
        raw = work_dir / f"speech-{index:03d}-raw.wav"
        fitted = work_dir / f"speech-{index:03d}.wav"
        _synthesize_live(segment.hindi_text, raw, settings)
        next_start = dialogue[index].start if index < len(dialogue) else segment.end
        available = max(segment.end - segment.start, next_start - segment.start - 0.05)
        _fit_duration(raw, fitted, available)
        inputs.extend(["-i", str(fitted)])
        delay_ms = max(0, round(segment.start * 1000))
        label = f"voice{index}"
        filters.append(
            f"[{index}:a]volume={settings.dialogue_volume:.3f},"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_labels.append(f"[{label}]")

    synthesis_finished = time.monotonic()
    filters.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=first:normalize=0,alimiter=limit=0.95[out]"
    )
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_path),
        ]
    )
    return {
        "separation": 0.0,
        "tts": round(synthesis_finished - started, 3),
        "mixing": round(time.monotonic() - synthesis_finished, 3),
    }
