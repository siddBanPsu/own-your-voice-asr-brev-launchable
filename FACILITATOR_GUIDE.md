# Facilitator guide

## Before the event

- Create the Brev Launchable from `launchable/` and rehearse the exact image on
  the intended L4 and A100 profiles.
- Confirm Python 3.12, NeMo 2.7.3, CUDA, Docker, and the NVIDIA Container
  Toolkit from the workshop kernel. Confirm Lab 3 opens with the separate
  **Own Your Voice Riva Client** kernel.
- Confirm instructor and attendees have the required NVIDIA AI Enterprise/NGC
  access before Labs 1 and 3.
- Pre-cache the Speech NIM image/model, Parakeet `.nemo` checkpoint, FLEURS
  splits, Parakeet ASR NIM 3.1.0 image, and Python dependencies.
- Measure Lab 2 training, `.nemo` save, `riva-build`, and `riva-deploy` duration
  on the exact event GPU. First-run downloads and TensorRT engine generation
  can dominate the schedule.
- Keep one prewarmed instructor instance, the completed `.nemo` and RMIR
  artifacts, and a backup Launchable link.
- If demonstrating EKS, have a platform owner pre-provision a GPU-enabled
  cluster, S3 RMIR location, NGC secrets, and gRPC-safe access path. Do not create
  cloud infrastructure from attendee notebooks.

## Timing and checkpoints

### Lab 1: Speech NIM baseline (75-90 minutes)

Checkpoint: the 0.6B NIM is ready and each participant produces an HTTP
transcript plus latency, real-time factor, and throughput-x-realtime. Stop the
NIM before Lab 2.

### Lab 2: NeMo adaptation (75-120 minutes)

Checkpoint: participants can explain the NeMo manifest, show zero unknown
tokens with the reused tokenizer, identify trainable layers, select the best
checkpoint by `val_wer`, compare untouched versus selected held-out WER, inspect
the English guardrail, and save `artifacts/parakeet-ctc-0.6b-nl.nemo`.

If validation or test does not improve, report it. Do not tune after repeatedly
viewing test results and do not manufacture an improvement for the workshop.

### Lab 3: Riva build and deployment (60-90 minutes)

Kernel checkpoint: Lab 3 must use **Own Your Voice Riva Client**. NeMo 2.7.3
and Riva client 2.26.0 are intentionally isolated because their Protobuf pins
are incompatible.

Checkpoint: `riva-build` creates `artifacts/riva/own_your_voice_asr.rmir`, the
local `riva-deploy` produces a GPU-optimized model repository, Riva becomes
ready on port 50051, and the Python client returns a held-out transcript.
Participants should be able to explain that Riva owns the optimized Triton
repository and that applications use the Riva gRPC API.

For the EKS discussion, map the same RMIR to the chart's `/data/rmir/<name>_v<version>/model.rmir`
layout, GPU-specific model generation, shared storage or S3 model cache,
homogeneous GPU nodes, HTTP/2 ingress, TLS, observability, and load testing.

## Recovery paths

- **NGC pull denied:** verify entitlement, Catalog access, and key validity.
- **NIM/Riva port conflict:** stop the Lab 1 NIM before Lab 3; both use 50051.
- **CUDA unavailable:** select **Own Your Voice ASR Labs** and rerun preflight.
- **Driver is too old for PyTorch:** confirm setup installed the `cu126` build
  and that `torch.version.cuda` reports `12.6`; do not use the default CUDA 13
  PyPI build on an image whose driver reports CUDA 12.7 capability.
- **NeMo import fails:** confirm Python 3.12 and rerun the pinned setup script;
  do not mix the archived tutorial's NeMo 1.23 install into this environment.
- **FLEURS download is slow:** use the instructor cache or reduce only the
  configurable sample counts; preserve official split boundaries.
- **Tokenizer audit fails:** inspect normalization and affected examples. Do
  not train through unknown labels or silently retrain a tokenizer mid-lab.
- **Out of memory in Lab 2:** stop NIM/Riva, set `LAB_PROFILE=t4`, restart the
  kernel, and train the CTC decoder only.
- **No `val_wer` checkpoint:** inspect NeMo validation logs and manifest paths;
  checkpoint selection must monitor `val_wer` with mode `min`.
- **RMIR build fails:** confirm the full `.nemo` artifact, Parakeet ASR NIM 3.1.0 image,
  NGC access, free disk, and integrated `nemo2riva` log output.
- **Riva deployment is slow:** engine generation is target-GPU work. Use the
  instructor's pre-generated repository only on the same GPU product.
- **EKS pod stays initializing:** inspect `riva-model-init`, PVC mounts, RMIR
  directory/version naming, NGC secrets, GPU allocation, and startup probes.
- **EKS scaling is slow:** use the Riva chart's shared or S3 model cache only
  across homogeneous GPU nodes, then load-test the real concurrency target.
