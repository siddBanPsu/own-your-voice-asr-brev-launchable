# Official source alignment

The workshop follows this NVIDIA stack boundary:

1. Speech NIM provides the supported Parakeet 0.6B baseline service.
2. NVIDIA NeMo performs model loading, fine-tuning, validation, and `.nemo`
   checkpoint export.
3. NVIDIA Riva ServiceMaker builds an RMIR from the NeMo model.
4. Riva deploys and optimizes its Triton model repository locally or through
   the Riva Helm chart on Amazon EKS.
5. Applications use the Riva gRPC API rather than treating internal Triton
   model inputs as the public speech contract.

## Training and model references

- [NVIDIA Riva Parakeet fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-parakeet-nemo.ipynb)
- [Parakeet CTC 0.6B model and `.nemo` artifact](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [NeMo ASR checkpoints and model classes](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html)
- [NeMo ASR configuration and fine-tuning](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html)
- [NeMo installation](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
- [Google FLEURS dataset and official language splits](https://huggingface.co/datasets/google/fleurs)

The reference notebook uses an older NeMo 1.23 environment and standalone
`nemo2riva`. This repository keeps its training pattern and same-tokenizer
recommendation but pins NeMo 2.7.3 for Python 3.12 and uses current Riva's
integrated NeMo conversion in `riva-build`.

## Riva build, local deployment, and client

- [Riva model development, `riva-build`, and `riva-deploy`](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/model-overview.html)
- [Building and deploying ASR pipelines](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-pipeline-configuration.html)
- [Riva support matrix, including Parakeet CTC 0.6B](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix/support-matrix.html)
- [Riva local Docker deployment](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/installation/deploy-local.html)
- [Riva Python client](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/apis/development-python.html)

## EKS and Kubernetes deployment

- [Riva Helm deployment and custom RMIR layout](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/installation/deploy-kubernetes.html)
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
