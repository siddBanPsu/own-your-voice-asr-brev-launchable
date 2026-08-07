from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .audio import duration_seconds, normalize_latin_text
from .profiles import LabProfile


DEFAULT_MODEL_ID = "nvidia/parakeet-ctc-0.6b"


def cuda_dtype():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these labs.")
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_processor(model_id: str = DEFAULT_MODEL_ID, training: bool = False):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    dtype = cuda_dtype()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id, dtype=dtype).to("cuda")
    model.train(training)
    if not training:
        model.eval()
    torch.cuda.empty_cache()
    return model, processor, dtype


def processor_inputs(processor, audio: Any, sample_rate: int, *, text: Any | None = None):
    kwargs: dict[str, Any] = {
        "audio": audio,
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
        "padding": True,
    }
    if text is not None:
        kwargs["text"] = text
    batch = processor(**kwargs)
    moved = {}
    for key, value in batch.items():
        if key == "input_features":
            moved[key] = value.to(device="cuda", dtype=cuda_dtype())
        else:
            moved[key] = value.to(device="cuda")
    return moved


def transcribe(model, processor, audio: np.ndarray, sample_rate: int) -> str:
    import torch

    inputs = processor_inputs(processor, audio, sample_rate)
    with torch.inference_mode():
        token_ids = model.generate(**inputs)
    return processor.batch_decode(token_ids, skip_special_tokens=True)[0]


def benchmark(model, processor, audio: np.ndarray, sample_rate: int, repeats: int = 5) -> dict[str, float]:
    import torch

    inputs = processor_inputs(processor, audio, sample_rate)
    with torch.inference_mode():
        model.generate(**inputs)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    audio_seconds = duration_seconds(audio, sample_rate)
    median = float(np.median(latencies))
    return {
        "audio_seconds": audio_seconds,
        "median_latency_seconds": median,
        "real_time_factor": median / audio_seconds,
        "throughput_x_realtime": audio_seconds / median,
    }


def configure_trainable_parameters(model, profile: LabProfile) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if profile.trainable_encoder_layers == -1:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.ctc_head.parameters():
            parameter.requires_grad = True
        if profile.trainable_encoder_layers:
            for layer in model.encoder.layers[-profile.trainable_encoder_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total, "percent": round(100 * trainable / total, 3)}


def tokenizer_coverage(processor, texts: Iterable[str]) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    unknown_id = tokenizer.unk_token_id
    total_tokens = 0
    unknown_tokens = 0
    affected_examples: list[str] = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        total_tokens += len(token_ids)
        count = token_ids.count(unknown_id) if unknown_id is not None else 0
        unknown_tokens += count
        if count and len(affected_examples) < 5:
            affected_examples.append(text)
    return {
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_fraction": unknown_tokens / total_tokens if total_tokens else 0.0,
        "affected_examples": affected_examples,
    }


def _ensure_duration_safe(records: Iterable[dict[str, Any]], max_seconds: int) -> None:
    for record in records:
        seconds = duration_seconds(record["audio"], record["sampling_rate"])
        if seconds > max_seconds + 1e-6:
            raise ValueError(
                f"Record {record.get('id', '<unknown>')} is {seconds:.1f}s, above the "
                f"{max_seconds}s limit. Filter it instead of truncating audio with a full transcript."
            )


def evaluate_wer(model, processor, records: Iterable[dict[str, Any]], max_seconds: int) -> dict[str, Any]:
    from jiwer import cer, wer

    records = list(records)
    if not records:
        raise ValueError("Evaluation requires at least one record.")
    _ensure_duration_safe(records, max_seconds)
    references: list[str] = []
    predictions: list[str] = []
    for record in records:
        predictions.append(
            transcribe(model, processor, record["audio"], record["sampling_rate"])
        )
        references.append(record["text"])
    normalized_references = [normalize_latin_text(text) for text in references]
    normalized_predictions = [normalize_latin_text(text) for text in predictions]
    return {
        "wer": float(wer(normalized_references, normalized_predictions)),
        "cer": float(cer(normalized_references, normalized_predictions)),
        "references": references,
        "predictions": predictions,
    }


def _snapshot_trainable_state(model) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model, state: dict[str, Any]) -> None:
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected best-checkpoint keys: {unexpected}")


def _set_training_mode(model, profile: LabProfile) -> None:
    model.train()
    if profile.trainable_encoder_layers != -1:
        model.encoder.eval()


def fine_tune_with_validation(
    model,
    processor,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    profile: LabProfile,
    *,
    eval_every: int,
    baseline_metrics: dict[str, Any] | None = None,
    seed: int = 7,
    gradient_clip_norm: float = 1.0,
) -> dict[str, Any]:
    import torch

    if not train_records:
        raise ValueError("Training requires at least one record.")
    if eval_every < 1:
        raise ValueError("eval_every must be positive.")
    if profile.train_batch_size > len(train_records):
        raise ValueError("Training batch size cannot exceed the training record count.")
    _ensure_duration_safe(train_records, profile.max_audio_seconds)
    _ensure_duration_safe(validation_records, profile.max_audio_seconds)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Run configure_trainable_parameters first.")
    optimizer = torch.optim.AdamW(trainable, lr=profile.learning_rate)
    use_scaler = cuda_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if profile.gradient_checkpointing and profile.trainable_encoder_layers == -1:
        model.gradient_checkpointing_enable()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    order = list(range(len(train_records)))
    rng.shuffle(order)
    cursor = 0

    baseline = baseline_metrics or evaluate_wer(
        model, processor, validation_records, profile.max_audio_seconds
    )
    best_step = 0
    best_wer = baseline["wer"]
    best_cer = baseline["cer"]
    best_state = _snapshot_trainable_state(model)
    losses: list[float] = []
    validation_history = [
        {"step": 0, "wer": best_wer, "cer": best_cer}
    ]

    _set_training_mode(model, profile)
    for step in range(1, profile.train_steps + 1):
        if cursor + profile.train_batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + profile.train_batch_size]
        cursor += profile.train_batch_size
        batch_records = [train_records[index] for index in indices]
        sample_rates = {record["sampling_rate"] for record in batch_records}
        if len(sample_rates) != 1:
            raise ValueError("All records in a training batch must share one sample rate.")
        inputs = processor_inputs(
            processor,
            [record["audio"] for record in batch_records],
            sample_rates.pop(),
            text=[record["text"] for record in batch_records],
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=cuda_dtype()):
            loss = model(**inputs).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

        if step % eval_every == 0 or step == profile.train_steps:
            model.eval()
            metrics = evaluate_wer(
                model, processor, validation_records, profile.max_audio_seconds
            )
            validation_history.append(
                {"step": step, "wer": metrics["wer"], "cer": metrics["cer"]}
            )
            print(
                f"step={step} loss={losses[-1]:.4f} "
                f"validation_wer={metrics['wer']:.4f} validation_cer={metrics['cer']:.4f}"
            )
            if (metrics["wer"], metrics["cer"]) < (best_wer, best_cer):
                best_step = step
                best_wer = metrics["wer"]
                best_cer = metrics["cer"]
                best_state = _snapshot_trainable_state(model)
            if step != profile.train_steps:
                _set_training_mode(model, profile)

    _restore_trainable_state(model, best_state)
    model.eval()
    return {
        "losses": losses,
        "validation_history": validation_history,
        "best_step": best_step,
        "best_wer": best_wer,
        "best_cer": best_cer,
    }


def save_trainable_state(model, path: str | Path) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(state, destination)
    return destination


def load_trainable_state(model, path: str | Path) -> list[str]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return list(missing)
