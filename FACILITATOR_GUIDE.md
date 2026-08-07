# Facilitator guide

## Before the event

- Publish the repository and create the Brev Launchable using the recorded
  settings under `launchable/`.
- Rehearse on the default AWS L4 profile and once on the intended A100 profile.
- Confirm attendees can see the Jupyter CTA, select the workshop kernel and
  download the model. The preflight output must report Python 3.12.
- Ask attendees to deploy 20-30 minutes before the lab. First-run downloads
  include a roughly 2.4 GB model and the Triton container.
- Keep one prewarmed instructor instance and one backup Launchable link.

## Timing and checkpoints

### Lab 1: deploy and benchmark (75-90 minutes)

Checkpoint: every participant produces a transcript plus latency, real-time
factor and throughput-x-realtime. Discuss why one short-file benchmark is not a
capacity plan.

### Lab 2: domain adaptation (75 minutes)

Checkpoint: participants can explain the train/validation boundary, see which
layers their GPU profile trains and save a small trainable-state checkpoint.
Do not present the tiny sample's WER movement as a customer result.

### Lab 3: ONNX and Triton (45 minutes)

Checkpoint: Triton reports ready, the HTTP client returns logits, and CTC
decoding yields a transcript. Close by mapping the local model repository to
EKS storage, a service endpoint, health checks, metrics and autoscaling.

## Recovery paths

- **Model download is slow:** pair participants with the prewarmed instance and
  continue the architecture discussion while caches populate.
- **CUDA is not available:** switch the notebook kernel to **Own Your Voice ASR
  Labs** and rerun `scripts/preflight.py`.
- **Out of memory in Lab 2:** set `LAB_PROFILE=t4`, restart the kernel and rerun
  the notebook. This freezes the encoder and trains only the CTC head.
- **Triton cannot start:** confirm Docker sees the GPU, verify that
  `model.onnx` exists, then inspect `docker logs own-your-voice-triton`.
- **ONNX export is slow:** use the instructor's exported model repository so
  participants can still complete the serving exercise.
