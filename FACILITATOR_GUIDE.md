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
- Measure Lab 2 training, `.nemo` save, `nemo2riva`, `riva-build`, and `riva-deploy` duration
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

Checkpoint: `nemo2riva` creates `artifacts/riva/own_your_voice_asr.riva`,
`riva-build` creates `artifacts/riva/own_your_voice_asr.rmir`, the
local `riva-deploy` produces a GPU-optimized model repository, Riva becomes
ready on port 50051, and the Python client returns a held-out transcript.
Participants should be able to explain that Riva owns the optimized Triton
repository and that applications use the Riva gRPC API.

Optional checkpoint: set `SAVE_INTERMEDIATE_ONNX = True` before the Lab 3
build to retain `artifacts/onnx/parakeet-ctc-0.6b-nl.onnx`. Budget for a second
export pass and enough disk space. The file is useful for graph inspection but
is not the input to `riva-build`; the `.riva` archive remains the supported
ServiceMaker input in this workshop.

For the EKS discussion, upload the same RMIR to S3 and map it through the Speech
NIM chart's `ngcModelConfigs`, then cover GPU-specific model generation,
persistent caches, homogeneous GPU nodes, HTTP/2 ingress, TLS, observability,
and load testing.

## Recovery paths

- **NGC pull denied:** verify entitlement, Catalog access, and key validity.
- **NIM/Riva port conflict:** stop the Lab 1 NIM before Lab 3; both use 50051.
- **Riva gRPC reports IPv6 `[::1]:50051`:** use `127.0.0.1:50051` as shown in
  Lab 3 so the local Docker endpoint uses IPv4 explicitly.
- **Empty Riva transcript or a crash in `GreedyDecoderSubword`:** rebuild the
  RMIR from the current script. Lab 3 uses TensorRT FP32 for the fine-tuned
  acoustic model so an all-blank result does not reach the timestamp decoder.
- **CUDA unavailable:** select **Own Your Voice ASR Labs** and rerun preflight.
- **L4 setup reports missing `sm_89` kernels:** update to the current setup
  script. CUDA 8.6 cubins can execute on compute capability 8.9, so an exact
  `torch.cuda.get_arch_list()` membership check is invalid. Current setup runs
  a real CUDA operation instead.
- **Driver is too old for PyTorch:** setup selects `cu126` for the documented
  L4/A100 paths and `cu129` for RTX PRO 6000 Blackwell (`sm_120`). Confirm that
  `torch.version.cuda` matches the selected backend. Do not use the default
  CUDA 13 PyPI build on an image whose driver reports CUDA 12.7 capability.
- **RTX PRO 6000 shows an `sm_120` PyTorch warning:** the existing environment
  still has the CUDA 12.6 wheel. Re-run Launchable setup, or use the repair
  command in the README, restart the kernel, and rerun `scripts/preflight.py`.
  Rehearse the NIM/Riva containers on this exact GPU before committing to it;
  TensorRT engine generation is target-GPU work.
- **Blackwell torch repair removes `pkg_resources`:** the broad uv `--reinstall`
  option refreshed torch's dependency graph and upgraded Setuptools. Restore
  `setuptools==80.9.0`; future repairs must use `--reinstall-package torch` and
  include the Setuptools pin as shown in the README.
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
- **Riva export fails:** confirm the full `.nemo` artifact, `nemo2riva==2.22.0`,
  ONNX opset 19, `max_dim=1000`, free disk, and the ONNX export log before
  debugging the container.
- **Optional ONNX copy is missing:** confirm the Lab 3 toggle is `True`, the
  NeMo environment exists at `~/.venvs/own-your-voice-asr`, and there is enough
  disk space. This does not block Riva when the toggle remains `False`.
- **RMIR build fails:** confirm the `.riva` artifact, Parakeet ASR NIM 3.1.0
  image, NGC access, and `riva-build speech_recognition` log output.
- **Riva deployment is slow:** engine generation is target-GPU work. Use the
  instructor's pre-generated repository only on the same GPU product.
- **EKS pod stays initializing:** inspect `riva-model-init`, PVC mounts, RMIR
  directory/version naming, NGC secrets, GPU allocation, and startup probes.
- **EKS scaling is slow:** use the Riva chart's shared or S3 model cache only
  across homogeneous GPU nodes, then load-test the real concurrency target.
