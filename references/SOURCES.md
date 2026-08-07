# Source alignment

The lab sequence follows the event brief:

1. Deploy Parakeet on the AWS G6/L4 target.
2. Fine-tune a Parakeet/Nemotron-family ASR model for domain speech.
3. Apply ONNX and NVIDIA Triton runtime patterns.

Implementation references:

- [NVIDIA Brev Launchables](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA Brev environments](https://docs.nvidia.com/brev/concepts/environments)
- [NVIDIA Parakeet CTC 0.6B model files](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [Parakeet architecture and Transformers API](https://huggingface.co/docs/transformers/model_doc/parakeet)
- [NVIDIA NeMo Speech ASR overview](https://docs.nvidia.com/nemo/speech/nightly/asr/intro.html)
- [NVIDIA NeMo Speech fine-tuning](https://docs.nvidia.com/nemo/speech/nightly/asr/fine_tuning.html)
- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Triton model repository format](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [NVIDIA Speech NIM ASR model choices](https://docs.nvidia.com/nim/speech/latest/asr/)

The core labs use open Parakeet weights and do not require an NGC API key. The
production handoff explains where Speech NIM/Riva provides a supported,
optimized serving path.
