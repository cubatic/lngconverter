import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .models import TranslatedSegment
from .pipeline import PipelineError


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise PipelineError(result.stderr.strip() or "Audio processing failed")


@lru_cache(maxsize=1)
def _load_tts(model_name: str, device_name: str):
    try:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PipelineError("Install the TTS dependencies from the Colab notebook") from exc

    device = device_name
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    return model, prompt_tokenizer, description_tokenizer, device


def _synthesize(text: str, output_path: Path, settings: Settings) -> None:
    try:
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise PipelineError("Install soundfile for TTS audio output") from exc

    model, tokenizer, description_tokenizer, device = _load_tts(
        settings.tts_model, settings.tts_device
    )
    description = description_tokenizer(
        settings.tts_voice_description, return_tensors="pt"
    ).to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        generation = model.generate(
            input_ids=description.input_ids,
            attention_mask=description.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
    sf.write(output_path, generation.float().cpu().numpy().squeeze(), model.config.sampling_rate)


def _fit_duration(source: Path, output: Path, duration: float) -> None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_duration = max(float(probe.stdout.strip()), 0.01)
    # Never slow a natural delivery merely to fill silence, and avoid the
    # chipmunk effect caused by forcing a long translation into a tiny slot.
    tempo = min(max(source_duration / max(duration, 0.1), 1.0), 1.3)
    output_duration = max(duration, source_duration / tempo)
    filters: list[str] = []
    while tempo > 2:
        filters.append("atempo=2")
        tempo /= 2
    filters.extend(
        [
            f"atempo={tempo:.6f}",
            f"apad=whole_dur={output_duration:.3f}",
            f"atrim=0:{output_duration:.3f}",
        ]
    )
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            ",".join(filters),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output),
        ]
    )


def _separate_background(source_audio: Path, work_dir: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise PipelineError("FFmpeg is required for dubbing")
    _run(
        [
            "python",
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            "--out",
            str(work_dir / "demucs"),
            str(source_audio),
        ]
    )
    background = work_dir / "demucs" / "htdemucs" / source_audio.stem / "no_vocals.wav"
    if not background.is_file():
        raise PipelineError("Demucs did not produce the background stem")
    return background


def create_dubbed_audio(
    source_audio: Path,
    dialogue: list[TranslatedSegment],
    output_path: Path,
    settings: Settings,
) -> None:
    if not dialogue:
        raise PipelineError("No Hindi dialogue is available for TTS")
    work_dir = output_path.parent / "dub-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    background = (
        _separate_background(source_audio, work_dir)
        if settings.enable_source_separation
        else source_audio
    )
    inputs = ["-i", str(background)]
    filter_parts: list[str] = []
    filter_parts.append(f"[0:a]volume={settings.background_volume:.3f}[background]")
    mix_labels = ["[background]"]
    for index, segment in enumerate(dialogue, start=1):
        raw = work_dir / f"speech-{index:03d}-raw.wav"
        fitted = work_dir / f"speech-{index:03d}.wav"
        _synthesize(segment.hindi_text, raw, settings)
        next_start = dialogue[index].start if index < len(dialogue) else segment.end
        available = max(segment.end - segment.start, next_start - segment.start - 0.1)
        _fit_duration(raw, fitted, available)
        inputs.extend(["-i", str(fitted)])
        delay_ms = max(0, round(segment.start * 1000))
        label = f"voice{index}"
        filter_parts.append(
            f"[{index}:a]volume={settings.dialogue_volume:.3f},"
            f"acompressor=threshold=0.12:ratio=3:attack=5:release=80,"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_labels.append(f"[{label}]")
    filter_parts.append(
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
            ";".join(filter_parts),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
    )
