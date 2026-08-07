from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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

            source = value.get("path")
            if not source:
                raise ValueError("Byte-only audio records are not supported by this lab helper.")
            audio, sample_rate = sf.read(source, dtype="float32")
            return _mono_float32(audio), int(sample_rate)

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return _mono_float32(data), int(samples.sample_rate)

    raise TypeError(f"Unsupported audio value: {type(value).__name__}")


def truncate_audio(audio: np.ndarray, sample_rate: int, max_seconds: int) -> np.ndarray:
    return audio[: int(sample_rate * max_seconds)]


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(audio) / sample_rate)


def load_dummy_librispeech(limit: int = 12) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    records: list[dict[str, Any]] = []
    for item in dataset.select(range(min(limit, len(dataset)))):
        array, sample_rate = decode_dataset_audio(item["audio"])
        records.append({"audio": array, "sampling_rate": sample_rate, "text": item["text"]})
    return records


def total_duration(records: Iterable[dict[str, Any]]) -> float:
    return sum(duration_seconds(record["audio"], record["sampling_rate"]) for record in records)
