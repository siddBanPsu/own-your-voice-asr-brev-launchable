# Source alignment

The lab sequence follows the event brief:

1. Deploy Parakeet on the AWS G6/L4 target.
2. Fine-tune a Parakeet/Nemotron-family ASR model for Dutch speech with held-out evaluation.
3. Apply ONNX and NVIDIA Triton runtime patterns.

Implementation references:

- [NVIDIA Brev Launchables](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA Brev environments](https://docs.nvidia.com/brev/concepts/environments)
- [NVIDIA Parakeet CTC 0.6B model files](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [Google FLEURS dataset and official language splits](https://huggingface.co/datasets/google/fleurs)
- [Parakeet architecture and Transformers API](https://huggingface.co/docs/transformers/model_doc/parakeet)
- [NVIDIA NeMo Speech ASR overview](https://docs.nvidia.com/nemo/speech/nightly/asr/intro.html)
- [NVIDIA NeMo Speech fine-tuning](https://docs.nvidia.com/nemo/speech/nightly/asr/fine_tuning.html)
- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Triton model repository format](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [NVIDIA Speech NIM ASR model choices](https://docs.nvidia.com/nim/speech/latest/asr/)
- [Deploy Parakeet CTC English with Speech NIM](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html)
- [NVIDIA ASR NIM support matrix and Parakeet 0.6B profiles](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)
- [Speech NIM prerequisites](https://docs.nvidia.com/nim/speech/latest/get-started/prerequisites.html)
- [NGC access setup](https://docs.nvidia.com/nim/speech/latest/get-started/ngc-access-setup.html)
- [Speech NIM model caching](https://docs.nvidia.com/nim/speech/latest/deployment/docker/model-caching.html)

Lab 1 uses the licensed Speech NIM container and requires NGC access. Labs 2-3
use the open Parakeet weights for customization and ONNX/Triton learning. A
locally customized Dutch checkpoint is not automatically a supported NIM
package. Lab 2 keeps FLEURS train, validation, and test separate and reports an
English-forgetting guardrail.
