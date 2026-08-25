# Official source alignment

The workshop follows this NVIDIA stack boundary:

1. Speech NIM provides the supported Parakeet 0.6B baseline service.
2. NVIDIA NeMo performs model loading, fine-tuning, validation, and `.nemo`
   checkpoint export.
3. NVIDIA Riva ServiceMaker builds an RMIR from the NeMo model.
4. The matching Parakeet ASR NIM image deploys and optimizes its Triton model
   repository locally or through the Speech NIM Helm chart on Amazon EKS.
5. Applications use the Riva gRPC API rather than treating internal Triton
   model inputs as the public speech contract.

## Training and model references

- [NVIDIA Riva Parakeet fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-parakeet-nemo.ipynb)
- [Parakeet CTC 0.6B model and `.nemo` artifact](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [NeMo ASR checkpoints and model classes](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html)
- [NeMo ASR configuration and fine-tuning](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html)
- [Current NeMo Speech ASR fine-tuning guide](https://docs.nvidia.com/nemo/speech/nightly/asr/fine_tuning.html)
- [NGC NeMo Speech framework container 26.07.00](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo-speech/26.07.00)
- [Archived NeMo 24.12 consolidated container instructions](https://docs.nvidia.com/nemo-framework/user-guide/24.12/installation.html)
- [CUDA major-version compatibility ranges](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NeMo installation](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
- [Google FLEURS dataset and official language splits](https://huggingface.co/datasets/google/fleurs)

The reference notebook uses an older NeMo 1.23 environment. This repository
keeps its training pattern and same-tokenizer recommendation, pins NeMo 2.7.3
and `nemo2riva` 2.22.0 for Python 3.12, exports the Parakeet CTC checkpoint with
ONNX opset 19 and `max_dim=1000`, and then uses the ASR NIM image's positional
ServiceMaker CLI.

The optional containerized Lab 2 path pins `nvcr.io/nvidia/nemo:24.12`, the
official consolidated ASR-capable NeMo container from the CUDA 12.6 generation,
because the workshop host currently has driver 565.57.01. The current
`nvcr.io/nvidia/nemo-speech:26.07.00` image supports Parakeet training but uses
CUDA 13.2 and explicitly requires driver 595.58 or later. Neither image is a
NeMo Microservices Customizer endpoint. The notebook therefore launches a
bounded local Docker training job rather than calling a remote fine-tuning API.

## Riva build, custom NIM deployment, and client

- [Deploy custom ASR models as Speech NIM](https://docs.nvidia.com/nim/speech/latest/asr/customization/custom-deployment.html)
- [NVIDIA `nemo2riva` converter and Parakeet CTC command](https://github.com/nvidia-riva/nemo2riva)
- [ASR pipeline configuration for `riva-build`](https://docs.nvidia.com/nim/speech/latest/asr/customization/pipeline-configuration.html)
- [Riva support matrix and the x86 Speech NIM boundary](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix/support-matrix.html)
- [Parakeet 0.6B ASR NIM container](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/parakeet-0-6b-ctc-en-us/-)
- [Riva Python client](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/apis/development-python.html)

## EKS and Kubernetes deployment

- [Speech NIM Helm deployment and S3 RMIR loading](https://docs.nvidia.com/nim/speech/latest/deployment/helm.html)
- [Amazon EKS NVIDIA GPU device management](https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia.html)
- [Amazon EKS AI/ML compute guidance](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)

The EKS material assumes an existing cluster. GPU support, Region, quota,
capacity, storage, NGC entitlement, ingress, and load-balancer policy must be
verified for the actual AWS account before the workshop.

## Brev, NIM, and Jupyter

- [NVIDIA Brev Launchables](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA Brev Jupyter notebooks](https://docs.nvidia.com/brev/latest/guides/development-tools/jupyter-notebooks)
- [NVIDIA Speech NIM ASR model choices](https://docs.nvidia.com/nim/speech/latest/asr/)
- [Deploy Parakeet CTC English with Speech NIM](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html)
- [Speech NIM prerequisites](https://docs.nvidia.com/nim/speech/latest/get-started/prerequisites.html)
- [NGC access setup](https://docs.nvidia.com/nim/speech/latest/get-started/ngc-access-setup.html)
- [JupyterLab file-navigation URLs](https://jupyterlab.readthedocs.io/en/latest/user/urls.html)
- [Jupyter Server configuration](https://jupyter-server.readthedocs.io/en/stable/users/configuration.html)
