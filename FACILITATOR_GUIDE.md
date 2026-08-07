# Facilitator guide

## Before the event

- Publish the repository and create the Brev Launchable using the recorded
  settings under `launchable/`.
- Rehearse on the default AWS L4 profile and once on the intended A100 profile.
- Confirm the instructor and attendees have NVIDIA AI Enterprise entitlement
  plus personal NGC API keys with Catalog access before Lab 1.
- Confirm the Jupyter CTA opens Lab 0 directly, attendees can select the
  workshop kernel, and the model downloads. The preflight output must report
  Python 3.12.
- Ask attendees to deploy 20-30 minutes before the lab. First-run downloads
  include the Speech NIM container/model, open 0.6B weights, Dutch FLEURS, and
  the Triton image.
- Keep one prewarmed instructor instance and one backup Launchable link.

## Timing and checkpoints

### Lab 1: deploy and benchmark (75-90 minutes)

Checkpoint: the 0.6B NIM reports ready and every participant produces an HTTP
transcript plus latency, real-time factor and throughput-x-realtime. Discuss
why one local, single-client benchmark is not a capacity plan. Stop the NIM
before Lab 2 so it releases GPU memory.

### Lab 2: Dutch cross-language adaptation (75-100 minutes)

Checkpoint: participants can explain tokenizer coverage, see which layers their
GPU profile trains, identify the best validation step, compare the untouched
Dutch test result, and report the English-forgetting guardrail. Step 0 is a
valid selection when no update improves validation. Do not present the subset
result as a customer or multilingual-production claim.

### Lab 3: ONNX and Triton (45 minutes)

Checkpoint: Triton reports ready, every returned FP32 logit is finite, CTC
decoding yields a transcript, and the Triton transcript matches the in-process
PyTorch path. Describe this as the portable correctness baseline, not a fully
optimized production engine. Close by mapping the local model repository to
mixed-precision/TensorRT validation, EKS storage, a service endpoint, health
checks, metrics and autoscaling.

## Recovery paths

- **NIM image pull is denied:** verify NVIDIA AI Enterprise entitlement, NGC
  Catalog access, and that the personal key has not expired.
- **NIM is slow to become ready:** allow up to 30 minutes on first startup,
  then inspect `docker logs parakeet-0-6b-ctc-en-us`.
- **NIM is unavailable on the GPU:** switch to an L4, A10, A100, or another
  support-matrix GPU; T4 does not meet the compute capability requirement.
- **Model download is slow:** pair participants with the prewarmed instance and
  continue the architecture discussion while caches populate.
- **FLEURS download is slow:** use the instructor's populated Hugging Face cache
  or reduce the configurable sample counts; do not merge the official splits.
- **Tokenizer coverage fails:** inspect the listed examples and normalization.
  Do not train through `<unk>` labels or silently move validation into train.
- **Dutch validation does not improve:** keep the selected step at 0 and report
  the result honestly. Do not tune controls after inspecting the test split.
- **CUDA is not available:** switch the notebook kernel to **Own Your Voice ASR
  Labs** and rerun `scripts/preflight.py`.
- **Out of memory in Lab 2:** set `LAB_PROFILE=t4`, restart the kernel and rerun
  the notebook. Confirm the NIM container is stopped first. This freezes the
  encoder and trains only the CTC head.
- **Triton cannot start:** confirm Docker sees the GPU, verify that
  `model.onnx` exists, then inspect `docker logs own-your-voice-triton`.
- **Token IDs are all zero or the transcript is empty:** inspect the notebook's
  finite-logit report. Rerun the FP32 export and restart Triton; do not serve an
  older FP16 graph or cast BF16 tensors to NumPy before converting them in
  PyTorch.
- **ONNX export is slow:** use the instructor's exported model repository so
  participants can still complete the serving exercise.
