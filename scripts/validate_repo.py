#!/usr/bin/env python3
from __future__ import annotations

import compileall
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ".python-version",
    "README.md",
    "requirements-riva-client.txt",
    "launchable/setup.sh",
    "launchable/brev-launchable.yaml",
    "scripts/start_nim.sh",
    "scripts/stop_nim.sh",
    "scripts/build_riva_rmir.sh",
    "scripts/export_nemo_onnx.py",
    "scripts/run_nemo_speech_container_finetune.py",
    "scripts/start_riva.sh",
    "scripts/stop_riva.sh",
    "labs/00_start_here.ipynb",
    "labs/01_deploy_and_benchmark.ipynb",
    "labs/02_domain_adaptation.ipynb",
    "labs/02_containerized_domain_adaptation.ipynb",
    "labs/03_riva_deployment.ipynb",
    "deploy/eks/README.md",
    "deploy/eks/values-custom-rmir.yaml",
]


def main() -> int:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert python_version == "3.12"

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject_text

    setup_text = (ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    assert setup_text.startswith("#!/bin/bash\n")
    assert 'PYTHON_VERSION="3.12"' in setup_text
    assert 'TORCH_BACKEND="${LAB_TORCH_BACKEND:-auto}"' in setup_text
    assert 'TORCH_BACKEND="cu126"' in setup_text
    assert 'TORCH_BACKEND="cu129"' in setup_text
    assert '--torch-backend "${TORCH_BACKEND}"' in setup_text
    assert 'torch.version.cuda != expected_cuda' in setup_text
    assert 'torch.ones(1, device="cuda")' in setup_text
    assert 'device_arch not in torch.cuda.get_arch_list()' not in setup_text
    assert '"${UV_BIN}" venv --managed-python --clear --python "${PYTHON_VERSION}"' in setup_text
    assert 'c.ServerApp.default_url = "/lab/tree/labs/00_start_here.ipynb?reset"' in setup_text
    assert 'c.ServerApp.root_dir = str(_workshop_root)' in setup_text
    embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY", setup_text, flags=re.DOTALL)
    assert embedded_python, "setup.sh must contain embedded Python checks"
    for index, source in enumerate(embedded_python):
        compile(source, f"launchable/setup.sh:python-{index}", "exec")

    manifest_text = (ROOT / "launchable" / "brev-launchable.yaml").read_text(encoding="utf-8")
    assert "mode: VM" in manifest_text
    assert "gpu: 1x NVIDIA L4" in manifest_text
    assert "pytorch_cuda_backend: auto (cu126 for L4/A100, cu129 for sm_120)" in manifest_text
    assert "RTX PRO 6000 Blackwell Server Edition" in manifest_text
    assert "container_id: parakeet-0-6b-ctc-en-us" in manifest_text
    assert "bs=1,mode=ofl" in manifest_text
    assert "parakeet-0-6b-ctc-en-us:3.1.0" in manifest_text
    assert "riva-nim:1.1.0" in manifest_text
    assert 'python_client_version: "2.26.0"' in manifest_text
    assert "name: tensorboard" in manifest_text
    assert "port: 6006" in manifest_text
    assert "show_as_call_to_action: false" in manifest_text

    nim_start_text = (ROOT / "scripts" / "start_nim.sh").read_text(encoding="utf-8")
    assert nim_start_text.startswith("#!/bin/bash\n")
    assert "parakeet-0-6b-ctc-en-us" in nim_start_text
    assert "bs=1,mode=ofl" in nim_start_text
    assert "/v1/health/ready" in nim_start_text

    for notebook in sorted((ROOT / "labs").glob("*.ipynb")):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"], f"{notebook.name} contains no cells"
        notebook_python = payload.get("metadata", {}).get("language_info", {}).get("version")
        assert notebook_python == "3.12", f"{notebook.name} must declare Python 3.12"
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{notebook.name}:cell-{index}", "exec")

    domain_payload = json.loads(
        (ROOT / "labs" / "02_domain_adaptation.ipynb").read_text(encoding="utf-8")
    )
    domain_source = "".join(
        line for cell in domain_payload["cells"] for line in cell.get("source", [])
    )
    assert "LANGUAGE_CONFIG = 'nl_nl'" in domain_source
    assert "nemo_asr.models.ASRModel.from_pretrained" in domain_source
    assert "nemo_tokenizer_coverage" in domain_source
    assert "monitor='val_wer'" in domain_source
    assert "baseline_test" in domain_source
    assert "selected_english" in domain_source
    assert "model.save_to" in domain_source
    assert "deterministic='warn'" in domain_source
    assert "deterministic=True" not in domain_source
    assert "TRAIN_EXAMPLES = COMMON_TRAIN_EXAMPLES" in domain_source
    assert "VALIDATION_EXAMPLES = COMMON_VALIDATION_EXAMPLES" in domain_source
    assert "TEST_EXAMPLES = COMMON_TEST_EXAMPLES" in domain_source
    assert "MAX_AUDIO_SECONDS = COMMON_MAX_AUDIO_SECONDS" in domain_source
    assert "TRAINABLE_ENCODER_LAYERS = profile.trainable_encoder_layers" in domain_source
    assert "ACCUMULATE_GRAD_BATCHES = profile.gradient_accumulation_steps" in domain_source
    assert "'train_examples': len(train_records)" in domain_source
    assert "'actual_examples': actual_examples" in domain_source
    assert "'validation_wer_relative_improvement_percent'" in domain_source
    assert "'validation_cer_relative_improvement_percent'" in domain_source
    assert "'validation_cer_absolute_improvement_percentage_points'" in domain_source
    assert "'test_wer_relative_improvement_percent'" in domain_source
    assert "'test_wer_absolute_improvement_percentage_points'" in domain_source
    assert "'test_cer_relative_improvement_percent'" in domain_source
    assert "'test_cer_absolute_improvement_percentage_points'" in domain_source
    assert "'english_wer_absolute_change_percentage_points'" in domain_source
    assert "'english_cer_absolute_change_percentage_points'" in domain_source
    assert "ENABLE_TENSORBOARD = True" in domain_source
    assert "from lightning.pytorch.loggers import TensorBoardLogger" in domain_source
    assert "training_logger = False" in domain_source
    assert "if ENABLE_TENSORBOARD:" in domain_source
    assert "logger=training_logger" in domain_source
    assert "artifacts' / 'tensorboard'" in domain_source

    container_domain_payload = json.loads(
        (ROOT / "labs" / "02_containerized_domain_adaptation.ipynb").read_text(
            encoding="utf-8"
        )
    )
    container_domain_source = "".join(
        line
        for cell in container_domain_payload["cells"]
        for line in cell.get("source", [])
    )
    assert "NEMO_CONTAINER_IMAGE = 'nvcr.io/nvidia/nemo:24.12'" in container_domain_source
    assert "LATEST_NEMO_SPEECH_IMAGE = 'nvcr.io/nvidia/nemo-speech:26.07.00'" in container_domain_source
    assert "'nvcr.io/nvidia/nemo:24.12': '560.35.05'" in container_domain_source
    assert "'nvcr.io/nvidia/nemo-speech:26.07.00': '595.58'" in container_domain_source
    assert "PULL_IMAGE = True" in container_domain_source
    assert "LANGUAGE_CONFIG = 'nl_nl'" in container_domain_source
    assert "TRAIN_EXAMPLES = COMMON_TRAIN_EXAMPLES" in container_domain_source
    assert "VALIDATION_EXAMPLES = COMMON_VALIDATION_EXAMPLES" in container_domain_source
    assert "TEST_EXAMPLES = COMMON_TEST_EXAMPLES" in container_domain_source
    assert "MAX_AUDIO_SECONDS = COMMON_MAX_AUDIO_SECONDS" in container_domain_source
    assert "TRAINABLE_ENCODER_LAYERS = profile.trainable_encoder_layers" in container_domain_source
    assert "ACCUMULATE_GRAD_BATCHES = profile.gradient_accumulation_steps" in container_domain_source
    assert "run_nemo_speech_container_finetune.py" in container_domain_source
    assert "--gpus', 'device=0'" in container_domain_source
    assert "--password-stdin" in container_domain_source
    assert "TemporaryDirectory(prefix='nemo-container-docker-auth-')" in container_domain_source
    assert "cuda_smoke_command" in container_domain_source
    assert "nemo.collections.asr as nemo_asr" in container_domain_source
    assert "assert torch.cuda.is_available()" in container_domain_source
    assert "PYTHONUNBUFFERED=1" in container_domain_source
    assert "'python', '-u'" in container_domain_source
    assert "NGC_API_KEY" not in container_domain_source
    assert "lab2_container_run_summary.json" in container_domain_source
    container_kernelspec = container_domain_payload["metadata"]["kernelspec"]
    assert container_kernelspec["name"] == "own-your-voice-asr"

    container_worker_text = (
        ROOT / "scripts" / "run_nemo_speech_container_finetune.py"
    ).read_text(encoding="utf-8")
    assert "ngc_nemo_framework_container" in container_worker_text
    assert '"model_load_started"' in container_worker_text
    assert '"baseline_validation_started"' in container_worker_text
    assert '"training_started"' in container_worker_text
    assert '"selected_test_complete"' in container_worker_text
    assert 'monitor="val_wer"' in container_worker_text
    assert "configure_nemo_trainable_parameters" in container_worker_text
    assert "baseline_validation_cer" in container_worker_text
    assert "selected_test_cer" in container_worker_text
    assert "english_cer_absolute_change_percentage_points" in container_worker_text
    assert "parakeet-ctc-0.6b-nl-container.nemo" in container_worker_text

    riva_payload = json.loads(
        (ROOT / "labs" / "03_riva_deployment.ipynb").read_text(encoding="utf-8")
    )
    riva_source = "".join(
        line for cell in riva_payload["cells"] for line in cell.get("source", [])
    )
    assert "ASR_NIM_TAG = '3.1.0'" in riva_source
    assert "build_riva_rmir.sh" in riva_source
    assert "start_riva.sh" in riva_source
    assert "riva.client.ASRService" in riva_source
    assert "offline_recognize" in riva_source
    assert "RIVA_URI = '127.0.0.1:50051'" in riva_source
    assert "localhost:50051" not in riva_source
    assert "CHECK_EKS_PREREQUISITES = False" in riva_source
    assert "SAVE_INTERMEDIATE_ONNX = False" in riva_source
    assert "RIVA_MIN_FREE_GB = 20" in riva_source
    assert "artifacts' / 'onnx' / 'parakeet-ctc-0.6b-nl.onnx'" in riva_source
    assert "'SAVE_INTERMEDIATE_ONNX': '1' if SAVE_INTERMEDIATE_ONNX else '0'" in riva_source
    assert "'RIVA_MIN_FREE_GB': str(RIVA_MIN_FREE_GB)" in riva_source

    riva_build_text = (ROOT / "scripts" / "build_riva_rmir.sh").read_text(
        encoding="utf-8"
    )
    assert "--entrypoint riva-build" in riva_build_text
    assert "nvcr.io/nim/nvidia/" in riva_build_text
    assert "ASR_NIM_TAG:-3.1.0" in riva_build_text
    assert "NEMO2RIVA_BIN" in riva_build_text
    assert "nemo2riva" in riva_build_text
    assert "NEMO_MODEL" in riva_build_text
    assert "model.riva" in riva_build_text
    assert 'ONNX_OPSET="${ONNX_OPSET:-19}"' in riva_build_text
    assert 'RIVA_MIN_FREE_GB="${RIVA_MIN_FREE_GB:-20}"' in riva_build_text
    assert 'RIVA_EXPORT_TMP_DIR="${RIVA_EXPORT_TMP_DIR:-${OUTPUT_DIR}/tmp}"' in riva_build_text
    assert 'df -Pk "${OUTPUT_DIR}"' in riva_build_text
    assert 'TMPDIR="${RIVA_EXPORT_TMP_DIR}"' in riva_build_text
    assert '--onnx-opset "${ONNX_OPSET}"' in riva_build_text
    assert "--max-dim 1000" in riva_build_text
    assert "speech_recognition" in riva_build_text
    assert "--decoder_type=greedy" in riva_build_text
    assert "--endpointing.residue_blanks_at_start=-16" in riva_build_text
    assert "--return_separate_utterances=True" in riva_build_text
    assert "--return_separate_utterances=False" not in riva_build_text
    assert "--nn.use_trt_fp32" in riva_build_text
    assert "--nn.fp16_needs_obey_precision_pass" not in riva_build_text
    assert "--chunk_size" not in riva_build_text
    assert "--config-path" not in riva_build_text
    assert "source_path=" not in riva_build_text
    assert 'SAVE_INTERMEDIATE_ONNX="${SAVE_INTERMEDIATE_ONNX:-0}"' in riva_build_text
    assert "scripts/export_nemo_onnx.py" in riva_build_text
    assert 'test -s "${ONNX_MODEL}"' in riva_build_text

    onnx_export_text = (ROOT / "scripts" / "export_nemo_onnx.py").read_text(
        encoding="utf-8"
    )
    assert "nemo_asr.models.ASRModel.restore_from" in onnx_export_text
    assert "model.export(" in onnx_export_text
    assert "onnx_opset_version=args.opset" in onnx_export_text
    assert "continues to package its own graph through nemo2riva" in onnx_export_text

    riva_start_text = (ROOT / "scripts" / "start_riva.sh").read_text(
        encoding="utf-8"
    )
    assert "--entrypoint riva-deploy" in riva_start_text
    assert "custom_model.tar.gz" in riva_start_text
    assert "NIM_DISABLE_MODEL_DOWNLOAD=true" in riva_start_text
    assert "http://127.0.0.1:9000/v1/health/ready" in riva_start_text
    assert "--publish 9000:9000" in riva_start_text
    assert "--publish 50051:50051" in riva_start_text

    requirements_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "nemo_toolkit[asr]==2.7.3" in requirements_text
    assert "nemo2riva==2.22.0" in requirements_text
    assert "jiwer==3.1.0" in requirements_text
    assert "tensorboard==2.20.0" in requirements_text
    assert "setuptools==80.9.0" in requirements_text
    assert "nvidia-riva-client" not in requirements_text

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "~/.venvs/own-your-voice-asr/bin/tensorboard" in readme_text
    assert "python -m tensorboard" not in readme_text
    assert '--logdir "${PWD}/artifacts/tensorboard"' in readme_text
    assert "--logdir artifacts/tensorboard" not in readme_text
    assert "A `git pull`" in readme_text
    assert "--reinstall-package torch" in readme_text
    assert "--reinstall torch==2.13.0" not in readme_text
    assert "## Dutch FLEURS dataset subset" in readme_text
    assert "| Train | 2,918 | 400 | 13.7% |" in readme_text
    assert "| **Total** | **3,453** | **550** | **15.9%** |" in readme_text
    assert "1,600 sample exposures" in readme_text

    riva_requirements_text = (ROOT / "requirements-riva-client.txt").read_text(
        encoding="utf-8"
    )
    assert "nvidia-riva-client==2.26.0" in riva_requirements_text
    assert "jiwer==4.0.0" in riva_requirements_text
    assert 'RIVA_VENV_DIR="${HOME}/.venvs/own-your-voice-riva"' in setup_text
    assert 'RIVA_KERNEL_NAME="own-your-voice-riva"' in setup_text
    assert "nemo2riva==2.22.0" in setup_text
    assert "tensorboard==2.20.0" in setup_text
    assert setup_text.count("setuptools==80.9.0") >= 2
    assert 'find_spec("pkg_resources")' in setup_text
    assert "appdirs==1.4.4" in setup_text
    assert "--no-build-isolation-package nvidia-pyindex" in setup_text

    preflight_text = (ROOT / "scripts" / "preflight.py").read_text(encoding="utf-8")
    assert '"nemo-toolkit"' in preflight_text
    assert '"nemo2riva"' in preflight_text
    assert '"tensorboard"' in preflight_text
    assert '"setuptools"' in preflight_text
    assert 'setuptools_version != "80.9.0"' in preflight_text
    assert 'find_spec("pkg_resources")' in preflight_text
    assert 'checks["cuda_smoke_test"] = "passed"' in preflight_text
    assert '"tritonclient"' not in preflight_text

    riva_kernelspec = riva_payload["metadata"]["kernelspec"]
    assert riva_kernelspec["name"] == "own-your-voice-riva"
    assert riva_kernelspec["display_name"] == "Own Your Voice Riva Client"

    for script in sorted((ROOT / "scripts").glob("*.sh")) + [ROOT / "launchable" / "setup.sh"]:
        subprocess.run(["bash", "-n", str(script)], check=True)

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        raise RuntimeError("Python source compilation failed")
    if not compileall.compile_file(ROOT / "scripts" / "export_nemo_onnx.py", quiet=1):
        raise RuntimeError("ONNX export helper compilation failed")
    if not compileall.compile_file(
        ROOT / "scripts" / "run_nemo_speech_container_finetune.py", quiet=1
    ):
        raise RuntimeError("Containerized NeMo Speech worker compilation failed")

    print("Repository structure, Python 3.12, NeMo/Riva contracts, notebooks, shell syntax and Python syntax are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
