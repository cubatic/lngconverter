from functools import lru_cache
from threading import Event

from .config import Settings
from .models import TranscriptSegment, TranslatedSegment
from .pipeline import PipelineCancelled, PipelineError

LANGUAGE_CODES = {
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "hi": "hin_Deva",
}


@lru_cache(maxsize=2)
def _load_model(model_name: str, source_code: str, device: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise PipelineError(
            "Install translation dependencies: pip install -e '.[translation]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def translate_to_hindi(
    segments: list[TranscriptSegment],
    source_language: str,
    settings: Settings,
    cancel_event: Event | None = None,
) -> list[TranslatedSegment]:
    if not segments:
        return []
    source_code = LANGUAGE_CODES.get(source_language)
    if not source_code:
        raise PipelineError(f"Translation does not support source language: {source_language}")
    tokenizer, model, device = _load_model(
        settings.translation_model, source_code, settings.translation_device
    )
    target_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES["hi"])
    results: list[TranslatedSegment] = []
    for offset in range(0, len(segments), 8):
        if cancel_event and cancel_event.is_set():
            raise PipelineCancelled("Processing paused or cancelled")
        batch = segments[offset : offset + 8]
        encoded = tokenizer(
            [segment.text for segment in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_id,
            max_new_tokens=128,
            num_beams=4,
        )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(
            TranslatedSegment(
                start=segment.start,
                end=segment.end,
                source_text=segment.text,
                hindi_text=hindi.strip(),
            )
            for segment, hindi in zip(batch, translated, strict=True)
        )
    return results
