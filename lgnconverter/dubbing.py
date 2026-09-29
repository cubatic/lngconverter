import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

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


@dataclass(frozen=True)
class ProsodyProfile:
    pitch_hz: float
    energy_db: float
    energy_variation_db: float
    energy_change_db: float
    characters_per_second: float


def _synthesize(
    text: str, output_path: Path, settings: Settings, description: str | None = None
) -> None:
    try:
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise PipelineError("Install soundfile for TTS audio output") from exc

    model, tokenizer, description_tokenizer, device = _load_tts(
        settings.tts_model, settings.tts_device
    )
    description = description_tokenizer(
        description or settings.tts_voice_description, return_tensors="pt"
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


def _separate_background(source_audio: Path, work_dir: Path) -> tuple[Path, Path]:
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
    vocals = work_dir / "demucs" / "htdemucs" / source_audio.stem / "vocals.wav"
    if not background.is_file() or not vocals.is_file():
        raise PipelineError("Demucs did not produce the expected stems")
    return background, vocals


def _estimate_pitch(signal: np.ndarray, sample_rate: int) -> float:
    frame_size = max(round(sample_rate * 0.04), 256)
    hop = max(frame_size // 2, 1)
    low_lag = max(round(sample_rate / 400), 1)
    high_lag = min(round(sample_rate / 70), frame_size - 1)
    pitches: list[float] = []
    for start in range(0, max(len(signal) - frame_size, 0), hop):
        frame = signal[start : start + frame_size]
        frame = frame - np.mean(frame)
        if np.sqrt(np.mean(frame**2)) < 0.006:
            continue
        correlation = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if correlation[0] <= 0:
            continue
        window = correlation[low_lag : high_lag + 1]
        lag = low_lag + int(np.argmax(window))
        if correlation[lag] / correlation[0] >= 0.25:
            pitches.append(sample_rate / lag)
    return float(np.median(pitches)) if pitches else 0.0


def _analyze_prosody(
    vocals_path: Path, segment: TranslatedSegment
) -> ProsodyProfile:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError("Install soundfile for prosody analysis") from exc
    audio, sample_rate = sf.read(vocals_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    start = max(0, round(segment.start * sample_rate))
    end = min(len(mono), round(segment.end * sample_rate))
    signal = mono[start:end]
    if not len(signal):
        return ProsodyProfile(0, -60, 0, 0, 0)
    frame_size = max(round(sample_rate * 0.025), 1)
    frame_count = max(len(signal) // frame_size, 1)
    trimmed = signal[: frame_count * frame_size]
    frames = trimmed.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
    levels = 20 * np.log10(rms + 1e-8)
    third = max(len(levels) // 3, 1)
    duration = max(segment.end - segment.start, 0.1)
    return ProsodyProfile(
        pitch_hz=_estimate_pitch(signal, sample_rate),
        energy_db=float(np.median(levels)),
        energy_variation_db=float(np.percentile(levels, 90) - np.percentile(levels, 10)),
        energy_change_db=float(np.mean(levels[-third:]) - np.mean(levels[:third])),
        characters_per_second=len(segment.source_text.strip()) / duration,
    )


def _prosody_description(profile: ProsodyProfile, settings: Settings) -> str:
    pitch = (
        "a noticeably high pitch"
        if profile.pitch_hz >= 220
        else "a moderately high pitch"
        if profile.pitch_hz >= 170
        else "a balanced pitch"
    )
    pace = (
        "fast, urgent pacing"
        if profile.characters_per_second >= 5.5
        else "slow, deliberate pacing"
        if profile.characters_per_second <= 2.7
        else "a natural moderate pace"
    )
    if profile.energy_db >= -27 and profile.energy_variation_db >= 12:
        emotion = "high excitement with strong dynamic emphasis"
    elif profile.energy_db >= -27:
        emotion = "firm intensity and confident emphasis"
    elif profile.energy_variation_db >= 14 and profile.energy_change_db > 3:
        emotion = "a vulnerable, emotional tone that rises toward urgency"
    elif profile.energy_db <= -39:
        emotion = "a soft, restrained and slightly sad tone"
    else:
        emotion = "subtle emotion with natural changes in loudness"
    return (
        f"{settings.tts_voice_description} For this line she uses {pitch}, {pace}, and "
        f"{emotion}. Preserve human pauses and vary emphasis naturally within the sentence."
    )


def create_dubbed_audio(
    source_audio: Path,
    dialogue: list[TranslatedSegment],
    output_path: Path,
    settings: Settings,
) -> dict[str, float]:
    if not dialogue:
        raise PipelineError("No Hindi dialogue is available for TTS")
    work_dir = output_path.parent / "dub-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if settings.enable_source_separation:
        background, vocals = _separate_background(source_audio, work_dir)
    else:
        background = vocals = source_audio
    separation_finished = time.monotonic()
    inputs = ["-i", str(background)]
    filter_parts: list[str] = []
    filter_parts.append(f"[0:a]volume={settings.background_volume:.3f}[background]")
    mix_labels = ["[background]"]
    for index, segment in enumerate(dialogue, start=1):
        raw = work_dir / f"speech-{index:03d}-raw.wav"
        fitted = work_dir / f"speech-{index:03d}.wav"
        description = settings.tts_voice_description
        if settings.enable_prosody_prompts:
            profile = _analyze_prosody(vocals, segment)
            description = _prosody_description(profile, settings)
        _synthesize(
            segment.hindi_text,
            raw,
            settings,
            description,
        )
        next_start = dialogue[index].start if index < len(dialogue) else segment.end
        available = max(segment.end - segment.start, next_start - segment.start - 0.1)
        _fit_duration(raw, fitted, available)
        inputs.extend(["-i", str(fitted)])
        delay_ms = max(0, round(segment.start * 1000))
        label = f"voice{index}"
        filter_parts.append(
            f"[{index}:a]volume={settings.dialogue_volume:.3f},"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_labels.append(f"[{label}]")
    synthesis_finished = time.monotonic()
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
    finished = time.monotonic()
    return {
        "separation": round(separation_finished - started, 3),
        "tts": round(synthesis_finished - separation_finished, 3),
        "mixing": round(finished - synthesis_finished, 3),
    }
