from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


@dataclass(frozen=True)
class LabProfile:
    name: str
    min_vram_gb: float
    trainable_encoder_layers: int
    train_batch_size: int
    max_audio_seconds: int
    train_steps: int
    learning_rate: float
    gradient_checkpointing: bool
    description: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, LabProfile] = {
    "t4": LabProfile(
        name="t4",
        min_vram_gb=14,
        trainable_encoder_layers=0,
        train_batch_size=1,
        max_audio_seconds=6,
        train_steps=4,
        learning_rate=5e-4,
        gradient_checkpointing=False,
        description="16 GB GPUs: CTC-head adaptation and short audio windows.",
    ),
    "l4": LabProfile(
        name="l4",
        min_vram_gb=20,
        trainable_encoder_layers=2,
        train_batch_size=1,
        max_audio_seconds=10,
        train_steps=8,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        description="20-32 GB GPUs: CTC head plus the last two encoder blocks.",
    ),
    "a100": LabProfile(
        name="a100",
        min_vram_gb=35,
        trainable_encoder_layers=-1,
        train_batch_size=2,
        max_audio_seconds=15,
        train_steps=12,
        learning_rate=2e-5,
        gradient_checkpointing=True,
        description="40 GB+ GPUs: full-model workshop fine-tune.",
    ),
}


def profile_for_vram(vram_gb: float) -> LabProfile:
    if vram_gb >= PROFILES["a100"].min_vram_gb:
        return PROFILES["a100"]
    if vram_gb >= PROFILES["l4"].min_vram_gb:
        return PROFILES["l4"]
    if vram_gb >= PROFILES["t4"].min_vram_gb:
        return PROFILES["t4"]
    raise RuntimeError(
        f"This workshop needs at least 14 GB of GPU memory; detected {vram_gb:.1f} GB. "
        "Choose a T4 16 GB, L4 22/24 GB, A10 24 GB, or larger GPU."
    )


def detected_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python kernel.")
        properties = torch.cuda.get_device_properties(0)
        return properties.total_memory / 1024**3
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Run the Brev setup script first.") from exc


def detect_profile(force: str | None = None, vram_gb: float | None = None) -> LabProfile:
    requested = (force or os.getenv("LAB_PROFILE", "auto")).strip().lower()
    if requested != "auto":
        if requested not in PROFILES:
            valid = ", ".join(["auto", *PROFILES])
            raise ValueError(f"Unknown LAB_PROFILE={requested!r}. Choose one of: {valid}.")
        return PROFILES[requested]
    return profile_for_vram(vram_gb if vram_gb is not None else detected_vram_gb())


def profile_table() -> list[dict[str, Any]]:
    return [profile.as_dict() for profile in PROFILES.values()]
