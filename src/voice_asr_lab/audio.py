from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
import re
from typing import Any
import unicodedata

import numpy as np


def _mono_float32(array: Any) -> np.ndarray:
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 8 else 1)
    return np.ravel(audio).astype(np.float32, copy=False)


def decode_dataset_audio(value: Any) -> tuple[np.ndarray, int]:
    """Decode both legacy datasets Audio dictionaries and TorchCodec decoders."""
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return _mono_float32(value["array"]), int(value["sampling_rate"])
        if "bytes" in value or "path" in value:
            import soundfile as sf

            source = BytesIO(value["bytes"]) if value.get("bytes") else value.get("path")
            if not source:
                raise ValueError("Audio record has neither bytes nor a readable path.")
            audio, sample_rate = sf.read(source, dtype="float32")
            return _mono_float32(audio), int(sample_rate)

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return _mono_float32(data), int(samples.sample_rate)

    raise TypeError(f"Unsupported audio value: {type(value).__name__}")


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio) / sample_rate)


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 16_000,
) -> tuple[np.ndarray, int]:
    if sample_rate == target_sample_rate:
        return audio, sample_rate
    import librosa

    resampled = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=target_sample_rate,
    )
    return _mono_float32(resampled), target_sample_rate


def normalize_latin_text(text: str) -> str:
    """Map Latin-script transcripts to the base model's lower-case alphabet."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    alphabet_only = re.sub(r"[^a-z' ]+", " ", without_marks)
    return " ".join(alphabet_only.split())


def load_fleurs_records(
    language_config: str,
    split: str,
    *,
    limit: int,
    max_audio_seconds: int,
) -> list[dict[str, Any]]:
    """Load normalized, duration-safe records from one official FLEURS split."""
    from datasets import Audio, load_dataset

    if limit < 1:
        raise ValueError("FLEURS record limit must be positive.")
    dataset = load_dataset("google/fleurs", language_config, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    records: list[dict[str, Any]] = []
    for item in dataset:
        audio, sample_rate = decode_dataset_audio(item["audio"])
        audio, sample_rate = resample_audio(audio, sample_rate)
        if duration_seconds(audio, sample_rate) > max_audio_seconds:
            continue
        original_text = str(item["transcription"])
        text = normalize_latin_text(original_text)
        if not text:
            continue
        records.append(
            {
                "audio": audio,
                "sampling_rate": sample_rate,
                "text": text,
                "original_text": original_text,
                "id": str(item.get("id", len(records))),
                "split": split,
            }
        )
        if len(records) == limit:
            break
    if len(records) < limit:
        raise RuntimeError(
            f"Requested {limit} duration-safe FLEURS {split} records but found "
            f"only {len(records)}. Increase max_audio_seconds or lower the limit."
        )
    return records


def load_dummy_librispeech(limit: int = 12) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    records: list[dict[str, Any]] = []
    for item in dataset.select(range(min(limit, len(dataset)))):
        array, sample_rate = decode_dataset_audio(item["audio"])
        array, sample_rate = resample_audio(array, sample_rate)
        records.append({"audio": array, "sampling_rate": sample_rate, "text": item["text"]})
    return records


def total_duration(records: Iterable[dict[str, Any]]) -> float:
    return sum(duration_seconds(record["audio"], record["sampling_rate"]) for record in records)
