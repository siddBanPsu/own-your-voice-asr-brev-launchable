from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TRITON_MODEL_NAME = "parakeet_ctc"
TRITON_HTTP_URL = "localhost:8000"


class ParakeetOnnxWrapper:
    def __init__(self, model):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, input_features, attention_mask):
                return self.wrapped(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits

        self.module = _Wrapper(model)


def export_fp16_onnx(model, sample_inputs: dict[str, Any], destination: str | Path) -> Path:
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to(device="cuda", dtype=torch.float16)
    wrapper = ParakeetOnnxWrapper(model).module.eval()
    features = sample_inputs["input_features"].to(device="cuda", dtype=torch.float16)
    mask = sample_inputs["attention_mask"].to(device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (features, mask),
            str(destination),
            input_names=["input_features", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch", 1: "frames"},
                "attention_mask": {0: "batch", 1: "frames"},
                "logits": {0: "batch", 1: "encoded_frames"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    return destination


def triton_config(vocab_size: int) -> str:
    return f'''name: "{TRITON_MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 1
input [
  {{ name: "input_features" data_type: TYPE_FP16 dims: [ -1, 80 ] }},
  {{ name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }}
]
output [
  {{ name: "logits" data_type: TYPE_FP16 dims: [ -1, {vocab_size} ] }}
]
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
dynamic_batching {{ preferred_batch_size: [ 1 ] max_queue_delay_microseconds: 1000 }}
'''


def write_triton_config(path: str | Path, vocab_size: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(triton_config(vocab_size), encoding="utf-8")
    return destination


def infer_triton(input_features: np.ndarray, attention_mask: np.ndarray, url: str = TRITON_HTTP_URL):
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=url)
    feature_input = httpclient.InferInput("input_features", input_features.shape, "FP16")
    mask_input = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    feature_input.set_data_from_numpy(input_features.astype(np.float16, copy=False))
    mask_input.set_data_from_numpy(attention_mask.astype(np.int64, copy=False))
    response = client.infer(TRITON_MODEL_NAME, inputs=[feature_input, mask_input])
    return response.as_numpy("logits")
