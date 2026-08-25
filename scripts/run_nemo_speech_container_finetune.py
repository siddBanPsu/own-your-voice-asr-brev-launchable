#!/usr/bin/env python3
"""Run the bounded Lab 2 ASR adaptation job inside an NGC NeMo Speech container."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "root",
        "container_image",
        "model_id",
        "language_name",
        "language_config",
        "train_manifest",
        "validation_manifest",
        "test_manifest",
        "english_manifest",
        "requested_examples",
        "max_audio_seconds",
        "max_steps",
        "learning_rate",
        "trainable_encoder_layers",
        "train_batch_size",
        "eval_batch_size",
        "accumulate_grad_batches",
        "validation_every_examples",
        "random_seed",
        "enable_tensorboard",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Container job config is missing fields: {missing}")
    return config


def _relative_improvement(baseline: float, selected: float) -> tuple[float, float]:
    absolute = baseline - selected
    relative = absolute / baseline if baseline else 0.0
    return absolute, relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())

    root = Path(config["root"]).resolve()
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import nemo
    import nemo.collections.asr as nemo_asr
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf

    from voice_asr_lab.nemo import (
        configure_nemo_trainable_parameters,
        evaluate_nemo_manifest,
        nemo_tokenizer_coverage,
        read_nemo_manifest,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The NeMo Speech container cannot see CUDA. Check Docker's NVIDIA runtime, "
            "the host driver, and the container CUDA compatibility."
        )

    manifests = {
        name: Path(config[f"{name}_manifest"]).resolve()
        for name in ("train", "validation", "test", "english")
    }
    missing_manifests = [str(path) for path in manifests.values() if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(f"Mounted manifests are missing: {missing_manifests}")

    train_batch_size = int(config["train_batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    max_steps = int(config["max_steps"])
    validation_every_examples = int(config["validation_every_examples"])
    val_check_interval = validation_every_examples // train_batch_size
    if validation_every_examples % train_batch_size:
        raise ValueError("validation_every_examples must be divisible by train_batch_size")
    if not 0 < val_check_interval <= max_steps:
        raise ValueError("The validation interval must be within the bounded training run")

    precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    seed_everything(int(config["random_seed"]), workers=True)
    device = torch.cuda.get_device_properties(0)
    runtime = {
        "execution_environment": "ngc_nemo_speech_container",
        "container_image": config["container_image"],
        "python": os.sys.version.split()[0],
        "nemo": getattr(nemo, "__version__", "unknown"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": device.name,
        "gpu_memory_gb": round(device.total_memory / 1024**3, 1),
        "precision": precision,
    }
    try:
        runtime["lightning"] = importlib.metadata.version("lightning")
    except importlib.metadata.PackageNotFoundError:
        runtime["lightning"] = "unknown"
    print({"container_runtime": runtime})

    model = nemo_asr.models.ASRModel.from_pretrained(
        config["model_id"], map_location="cuda"
    ).to("cuda")
    transcript_rows = [
        row
        for split in ("train", "validation", "test")
        for row in read_nemo_manifest(manifests[split])
    ]
    coverage = nemo_tokenizer_coverage(
        model.tokenizer, [row["text"] for row in transcript_rows]
    )
    print({"model_class": type(model).__name__, "tokenizer_coverage": coverage})
    if coverage["unknown_tokens"]:
        raise RuntimeError("Tokenizer coverage failed before container training.")

    model.eval()
    baseline_validation = evaluate_nemo_manifest(
        model, manifests["validation"], eval_batch_size
    )
    baseline_test = evaluate_nemo_manifest(model, manifests["test"], eval_batch_size)
    baseline_english = evaluate_nemo_manifest(
        model, manifests["english"], eval_batch_size
    )
    print(
        {
            "baseline_validation_wer": baseline_validation["wer"],
            "baseline_validation_cer": baseline_validation["cer"],
            "baseline_test_wer": baseline_test["wer"],
            "baseline_test_cer": baseline_test["cer"],
            "baseline_english_wer": baseline_english["wer"],
            "baseline_english_cer": baseline_english["cer"],
        }
    )

    parameter_summary = configure_nemo_trainable_parameters(
        model, int(config["trainable_encoder_layers"])
    )
    checkpoint_dir = root / "artifacts" / "nemo_container_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="parakeet-nl-container-{step:04d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    training_logger: TensorBoardLogger | bool = False
    if bool(config["enable_tensorboard"]):
        training_logger = TensorBoardLogger(
            save_dir=str(root / "artifacts" / "tensorboard"),
            name="parakeet-nl-container",
            default_hp_metric=False,
        )
        print({"tensorboard_log_dir": training_logger.log_dir})

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        max_steps=max_steps,
        max_epochs=-1,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=int(config["accumulate_grad_batches"]),
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        logger=training_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        deterministic="warn",
    )
    model.set_trainer(trainer)
    model.setup_training_data(
        train_data_config={
            "manifest_filepath": str(manifests["train"]),
            "sample_rate": 16000,
            "batch_size": train_batch_size,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
        }
    )
    model.setup_validation_data(
        val_data_config={
            "manifest_filepath": str(manifests["validation"]),
            "sample_rate": 16000,
            "batch_size": eval_batch_size,
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": True,
        }
    )
    learning_rate = float(config["learning_rate"])
    model.setup_optimization(
        optim_config=OmegaConf.create(
            {
                "name": "adamw",
                "lr": learning_rate,
                "betas": [0.9, 0.98],
                "weight_decay": 0.001,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": min(25, max_steps // 10),
                    "min_lr": learning_rate / 20,
                    "max_steps": max_steps,
                },
            }
        )
    )
    print({"trainable_parameters": parameter_summary})
    trainer.fit(model)

    best_checkpoint = Path(checkpoint_callback.best_model_path)
    if not best_checkpoint.is_file():
        raise RuntimeError("No best container checkpoint was saved; inspect val_wer.")
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to("cuda").eval()
    nemo_artifact = root / "artifacts" / "parakeet-ctc-0.6b-nl-container.nemo"
    model.save_to(str(nemo_artifact))

    selected_validation = evaluate_nemo_manifest(
        model, manifests["validation"], eval_batch_size
    )
    selected_test = evaluate_nemo_manifest(model, manifests["test"], eval_batch_size)
    selected_english = evaluate_nemo_manifest(
        model, manifests["english"], eval_batch_size
    )
    validation_wer_absolute, validation_wer_relative = _relative_improvement(
        baseline_validation["wer"], selected_validation["wer"]
    )
    validation_cer_absolute, validation_cer_relative = _relative_improvement(
        baseline_validation["cer"], selected_validation["cer"]
    )
    test_wer_absolute, test_wer_relative = _relative_improvement(
        baseline_test["wer"], selected_test["wer"]
    )
    test_cer_absolute, test_cer_relative = _relative_improvement(
        baseline_test["cer"], selected_test["cer"]
    )
    actual_examples = {
        split: len(read_nemo_manifest(manifests[split]))
        for split in ("train", "validation", "test")
    }
    effective_batch_size = train_batch_size * int(config["accumulate_grad_batches"])
    sample_exposures = effective_batch_size * max_steps
    summary = {
        **runtime,
        "model_id": config["model_id"],
        "language": config["language_name"],
        "language_config": config["language_config"],
        "best_checkpoint": str(best_checkpoint),
        "best_validation_wer": float(checkpoint_callback.best_model_score),
        "baseline_validation_wer": baseline_validation["wer"],
        "selected_validation_wer": selected_validation["wer"],
        "validation_wer_absolute_improvement": validation_wer_absolute,
        "validation_wer_absolute_improvement_percentage_points": round(validation_wer_absolute * 100, 2),
        "validation_wer_relative_improvement": validation_wer_relative,
        "validation_wer_relative_improvement_percent": round(validation_wer_relative * 100, 2),
        "baseline_validation_cer": baseline_validation["cer"],
        "selected_validation_cer": selected_validation["cer"],
        "validation_cer_absolute_improvement": validation_cer_absolute,
        "validation_cer_absolute_improvement_percentage_points": round(validation_cer_absolute * 100, 2),
        "validation_cer_relative_improvement": validation_cer_relative,
        "validation_cer_relative_improvement_percent": round(validation_cer_relative * 100, 2),
        "baseline_test_wer": baseline_test["wer"],
        "selected_test_wer": selected_test["wer"],
        "test_wer_absolute_improvement": test_wer_absolute,
        "test_wer_absolute_improvement_percentage_points": round(test_wer_absolute * 100, 2),
        "test_wer_relative_improvement": test_wer_relative,
        "test_wer_relative_improvement_percent": round(test_wer_relative * 100, 2),
        "baseline_test_cer": baseline_test["cer"],
        "selected_test_cer": selected_test["cer"],
        "test_cer_absolute_improvement": test_cer_absolute,
        "test_cer_absolute_improvement_percentage_points": round(test_cer_absolute * 100, 2),
        "test_cer_relative_improvement": test_cer_relative,
        "test_cer_relative_improvement_percent": round(test_cer_relative * 100, 2),
        "baseline_english_wer": baseline_english["wer"],
        "selected_english_wer": selected_english["wer"],
        "english_wer_absolute_change_percentage_points": round((selected_english["wer"] - baseline_english["wer"]) * 100, 2),
        "baseline_english_cer": baseline_english["cer"],
        "selected_english_cer": selected_english["cer"],
        "english_cer_absolute_change_percentage_points": round((selected_english["cer"] - baseline_english["cer"]) * 100, 2),
        "nemo_artifact": str(nemo_artifact),
        "tokenizer_retrained": False,
        "tokenizer_coverage": coverage,
        "max_steps": max_steps,
        "max_audio_seconds": int(config["max_audio_seconds"]),
        "requested_examples": config["requested_examples"],
        "actual_examples": actual_examples,
        "effective_batch_size": effective_batch_size,
        "sample_exposures": sample_exposures,
        "effective_training_passes": sample_exposures / actual_examples["train"],
        "trainable_encoder_layers": int(config["trainable_encoder_layers"]),
        "trainable_parameters": parameter_summary,
    }
    summary_path = root / "artifacts" / "lab2_container_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    changed = 0
    for reference, before, after in zip(
        baseline_test["references"],
        baseline_test["predictions"],
        selected_test["predictions"],
    ):
        if before != after:
            print(f"REF:    {reference}\nBEFORE: {before}\nAFTER:  {after}\n")
            changed += 1
        if changed == 5:
            break
    if changed == 0:
        print("No held-out transcript changed; report this outcome without tuning on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
