#!/usr/bin/env python3
"""Export a standalone ONNX copy from a complete NeMo ASR checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a standalone ONNX artifact for inspection. The Riva build "
            "continues to package its own graph through nemo2riva."
        )
    )
    parser.add_argument("--nemo-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def iter_paths(value: object) -> Iterator[Path]:
    """Yield path-like values from NeMo's version-dependent export result."""
    if isinstance(value, (str, os.PathLike)):
        yield Path(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_paths(item)


def main() -> int:
    args = parse_args()
    nemo_model = args.nemo_model.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not nemo_model.is_file():
        raise FileNotFoundError(f"NeMo checkpoint not found: {nemo_model}")
    if nemo_model.suffix != ".nemo":
        raise ValueError(f"Expected a .nemo checkpoint, received: {nemo_model}")
    if output.suffix != ".onnx":
        raise ValueError(f"ONNX output must end in .onnx, received: {output}")
    if args.opset < 1:
        raise ValueError("ONNX opset must be a positive integer.")

    import torch
    import nemo.collections.asr as nemo_asr

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested, but PyTorch cannot see a GPU.")
    device_name = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    device = torch.device(device_name)

    output.parent.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_model),
        map_location=device,
    )
    model.to(device)
    model.eval()
    export_result = model.export(
        output=str(output),
        onnx_opset_version=args.opset,
    )

    candidates = {output}
    for candidate in iter_paths(export_result):
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if resolved.suffix.lower() == ".onnx":
            candidates.add(resolved.resolve())
    candidates.update(output.parent.glob(f"{output.stem}*.onnx"))
    artifacts = sorted(
        path.resolve() for path in candidates if path.is_file() and path.stat().st_size > 0
    )
    if output not in artifacts:
        raise RuntimeError(f"NeMo export did not create the requested ONNX file: {output}")

    print(
        json.dumps(
            {
                "onnx": str(output),
                "size_mb": round(output.stat().st_size / 1024**2, 2),
                "opset": args.opset,
                "device": device_name,
                "exported_files": [str(path) for path in artifacts],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
