# Own Your Voice: Parakeet, NeMo, and Riva on NVIDIA Brev

A single-GPU workshop that connects NVIDIA's speech stack end to end:

```text
Parakeet Speech NIM baseline
        ↓
NeMo fine-tuning + held-out evaluation
        ↓
selected .nemo checkpoint
        ↓
Riva ServiceMaker → RMIR
        ↓
Riva API on EKS or local Docker
        ↓
TensorRT + Triton inside Riva
```

The repository is pinned to CPython 3.12. The Brev setup script installs its
own managed Python 3.12 runtime with `uv`, so Ubuntu's system Python is not used.

## Lab flow

| Lab | Outcome | Workshop time |
|---|---|---:|
| 0. Start here | Verify Python 3.12, CUDA, disk, Docker, and the GPU profile | 10 min |
| 1. Speech NIM | Deploy Parakeet CTC 0.6B and benchmark its supported API | 75-90 min |
| 2. NVIDIA NeMo | Fine-tune on Dutch FLEURS, select by `val_wer`, test once, and save a complete `.nemo` model | 75-120 min |
| 3. NVIDIA Riva | Build an RMIR, serve locally through Riva gRPC, and map the same artifact to Riva on EKS | 60-90 min |

The Dutch exercise demonstrates the customization workflow. A small cross-
language subset is not evidence of production Dutch accuracy.

## Why the tokenizer is reused

Lab 2 follows NVIDIA Riva's
[`asr-finetune-parakeet-nemo.ipynb`](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-parakeet-nemo.ipynb).
That notebook recommends reusing a pretrained tokenizer for adaptation datasets
below 50 hours. The lab therefore:

- keeps the tokenizer embedded in `nvidia/parakeet-ctc-0.6b`;
- normalizes Dutch text to the checkpoint's lower-case Latin output contract;
- audits every train, validation, and test transcript for unknown tokens; and
- stops before training if coverage is incomplete.

This preserves the decoder shape and simplifies Riva packaging. It does not
create a new native Dutch vocabulary or language model.

## GPU profiles

`LAB_PROFILE=auto` selects conservative settings from detected GPU memory.

| GPU class | Profile | NeMo behavior |
|---|---|---|
| T4 16 GB | `t4` | Labs 2-3 only; train the CTC decoder on a smaller subset |
| L4/A10 20-24 GB | `l4` | Train the decoder plus the final two encoder blocks |
| A100 40/80 GB | `a100` | Same safe default, with room to increase the controls |

Recommended Brev default: AWS `g6.2xlarge`, one L4, 32 GB host RAM, and at
least 150 GB disk. Recheck provider availability, price, image, and capacity
before the event. Stop the Lab 1 NIM before training or Riva deployment so only
one speech stack owns the GPU.

## Access requirements

Lab 1 and Lab 3 require NVIDIA NGC access and the applicable NVIDIA AI
Enterprise entitlement for the Speech NIM and Riva containers. Each notebook
asks for a personal NGC API key with hidden input. The scripts use a temporary
Docker configuration and do not add the key to the repository, notebook,
Launchable defaults, or long-lived Docker configuration.

Lab 2 uses the open Parakeet checkpoint with NVIDIA NeMo 2.7.3. Lab 3 pins the
Riva container, Helm chart, and client to 2.26.0.

## Lab 3 deployment choices

### Amazon EKS — production-oriented path

The RMIR from Lab 3 is staged in the Riva model volume and selected by the Riva
Helm chart. The chart runs target-GPU optimization, generates the Triton model
repository, starts the Riva API, and exposes a Kubernetes Service. See
[`deploy/eks/README.md`](deploy/eks/README.md).

The repository deliberately does not create an EKS cluster. A platform owner
must first confirm AWS account/Region, GPU node availability, EKS version and
GPU device mechanism, storage class, NGC secrets, DNS, gRPC ingress, TLS, and
cost. Model caches can be reused only across homogeneous GPU products.

### Brev/local Docker — runnable workshop path

The notebook calls:

```bash
bash scripts/build_riva_rmir.sh
bash scripts/start_riva.sh
```

`riva-deploy` optimizes the RMIR for the current GPU and Riva serves the model
on gRPC port 50051. Applications call Riva, not raw Triton. Stop it with:

```bash
bash scripts/stop_riva.sh
```

## Create the Brev Launchable

1. Push this repository to GitHub.
2. Follow [`launchable/README.md`](launchable/README.md).
3. Preview on the intended L4 and A100 profiles.
4. Run Labs 0-2 with **Own Your Voice ASR Labs**. Lab 3 records and selects the
   separate **Own Your Voice Riva Client** kernel automatically.
5. Share the Launchable link and ask attendees to deploy 20-30 minutes early.

The setup script starts with `#!/bin/bash`, installs managed Python 3.12, and
creates isolated NeMo and Riva client environments. This is required because
NeMo 2.7.3 and Riva client 2.26.0 pin incompatible Protobuf versions. Fresh
managed-Jupyter sessions open `labs/00_start_here.ipynb` directly.

## Run on an existing Brev instance

```bash
bash launchable/setup.sh
~/.venvs/own-your-voice-asr/bin/python scripts/preflight.py
```

## Validation boundary

Repository structure, Python and shell syntax, notebook code, dependency pins,
and deployment contracts can be validated locally with:

```bash
python scripts/validate_repo.py
python -m unittest discover -s tests
```

A real workshop release still requires a GPU rehearsal of the exact NeMo
version, Parakeet checkpoint, dataset size, Riva container, target GPU, RMIR
build time, engine-generation time, Riva transcript, and EKS chart/storage
configuration. Do not describe static checks as a live Riva or EKS deployment.

See [`references/SOURCES.md`](references/SOURCES.md) for official references.
