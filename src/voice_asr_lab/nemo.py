from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from .audio import duration_seconds, normalize_latin_text


def write_nemo_manifest(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    split_name: str,
) -> Path:
    """Write 16 kHz WAV files and a NeMo JSON-lines ASR manifest."""
    import soundfile as sf

    records = list(records)
    if not records:
        raise ValueError("A NeMo manifest requires at least one record.")

    destination = Path(output_dir)
    audio_dir = destination / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"{split_name}.jsonl"

    lines: list[str] = []
    for index, record in enumerate(records):
        sample_rate = int(record["sampling_rate"])
        if sample_rate != 16_000:
            raise ValueError(
                f"NeMo workshop audio must be 16 kHz; record {index} is {sample_rate} Hz."
            )
        text = normalize_latin_text(str(record["text"]))
        if not text:
            raise ValueError(f"Record {index} has an empty normalized transcript.")
        audio_path = (audio_dir / f"{split_name}-{index:05d}.wav").resolve()
        sf.write(audio_path, record["audio"], sample_rate, subtype="PCM_16")
        lines.append(
            json.dumps(
                {
                    "audio_filepath": str(audio_path),
                    "duration": duration_seconds(record["audio"], sample_rate),
                    "text": text,
                },
                ensure_ascii=True,
            )
        )

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def read_nemo_manifest(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unknown_token_id(tokenizer: Any) -> int | None:
    candidates = [tokenizer, getattr(tokenizer, "tokenizer", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "unk_id", None)
        if callable(value):
            value = value()
        if isinstance(value, int) and value >= 0:
            return value
        value = getattr(candidate, "unk_token_id", None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def nemo_tokenizer_coverage(tokenizer: Any, texts: Iterable[str]) -> dict[str, Any]:
    """Audit NeMo tokenizer coverage before reusing a pretrained tokenizer."""
    unknown_id = _unknown_token_id(tokenizer)
    total_tokens = 0
    unknown_tokens = 0
    affected_examples: list[str] = []
    for text in texts:
        token_ids = list(tokenizer.text_to_ids(text))
        total_tokens += len(token_ids)
        count = token_ids.count(unknown_id) if unknown_id is not None else 0
        unknown_tokens += count
        if count and len(affected_examples) < 5:
            affected_examples.append(text)
    return {
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_fraction": unknown_tokens / total_tokens if total_tokens else 0.0,
        "unknown_id": unknown_id,
        "affected_examples": affected_examples,
    }


def transcription_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, (list, tuple)) and item:
        return transcription_text(item[0])
    raise TypeError(f"Unsupported NeMo transcription result: {type(item).__name__}")


def evaluate_nemo_manifest(model: Any, manifest_path: str | Path, batch_size: int) -> dict[str, Any]:
    from jiwer import cer, wer

    rows = read_nemo_manifest(manifest_path)
    paths = [row["audio_filepath"] for row in rows]
    references = [normalize_latin_text(row["text"]) for row in rows]
    raw_predictions = model.transcribe(paths, batch_size=batch_size)
    predictions = [normalize_latin_text(transcription_text(item)) for item in raw_predictions]
    if len(predictions) != len(references):
        raise RuntimeError(
            f"NeMo returned {len(predictions)} transcripts for {len(references)} files."
        )
    return {
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
        "references": references,
        "predictions": predictions,
    }


def configure_nemo_trainable_parameters(model: Any, encoder_layers: int) -> dict[str, float | int]:
    """Freeze Parakeet, then unfreeze the CTC decoder and selected encoder tail."""
    if encoder_layers < -1:
        raise ValueError("encoder_layers must be -1 (all), 0, or a positive tail count.")
    for parameter in model.parameters():
        parameter.requires_grad = encoder_layers == -1

    if encoder_layers != -1:
        for parameter in model.decoder.parameters():
            parameter.requires_grad = True
        if encoder_layers:
            layers = list(model.encoder.layers)
            if encoder_layers > len(layers):
                raise ValueError(
                    f"Requested {encoder_layers} encoder layers, but model has {len(layers)}."
                )
            for layer in layers[-encoder_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percent": round(100 * trainable / total, 3),
    }
