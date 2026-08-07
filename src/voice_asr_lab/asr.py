from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .audio import duration_seconds, truncate_audio
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


def evaluate_wer(model, processor, records: Iterable[dict[str, Any]], max_seconds: int) -> dict[str, Any]:
    from jiwer import wer

    references: list[str] = []
    predictions: list[str] = []
    for record in records:
        audio = truncate_audio(record["audio"], record["sampling_rate"], max_seconds)
        predictions.append(transcribe(model, processor, audio, record["sampling_rate"]))
        references.append(record["text"])
    return {
        "wer": float(wer([text.lower() for text in references], [text.lower() for text in predictions])),
        "references": references,
        "predictions": predictions,
    }


def tiny_finetune(model, processor, records: list[dict[str, Any]], profile: LabProfile) -> list[float]:
    import torch

    random.seed(7)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Run configure_trainable_parameters first.")
    optimizer = torch.optim.AdamW(trainable, lr=profile.learning_rate)
    use_scaler = cuda_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model.train()
    if profile.trainable_encoder_layers != -1:
        model.encoder.eval()
    if profile.gradient_checkpointing and profile.trainable_encoder_layers == -1:
        model.gradient_checkpointing_enable()

    losses: list[float] = []
    for step in range(profile.train_steps):
        start = (step * profile.train_batch_size) % len(records)
        batch_records = [records[(start + offset) % len(records)] for offset in range(profile.train_batch_size)]
        sample_rates = {record["sampling_rate"] for record in batch_records}
        if len(sample_rates) != 1:
            raise ValueError("All records in a training batch must share one sample rate.")
        audio = [
            truncate_audio(record["audio"], record["sampling_rate"], profile.max_audio_seconds)
            for record in batch_records
        ]
        text = [record["text"] for record in batch_records]
        inputs = processor_inputs(
            processor,
            audio,
            sample_rates.pop(),
            text=text,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=cuda_dtype()):
            loss = model(**inputs).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


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
