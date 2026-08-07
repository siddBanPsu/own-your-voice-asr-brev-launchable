# Own Your Voice: Parakeet ASR Labs on NVIDIA Brev

A GitHub-ready, single-GPU lab environment for the AWS + NVIDIA roadshow. The
core path is credential-free and follows the event brief: deploy Parakeet,
adapt it to domain speech, then export it to ONNX and serve it with NVIDIA
Triton Inference Server.

## Lab flow

| Lab | Outcome | Workshop time |
|---|---|---:|
| 0. Start here | Verify CUDA, disk, Docker and the selected memory profile | 10 min |
| 1. Deploy and benchmark | Run Parakeet CTC 0.6B, measure latency and real-time factor | 75-90 min |
| 2. Domain adaptation | Establish a held-out baseline, run a memory-aware micro-fine-tune, remeasure WER | 75 min |
| 3. ONNX + Triton | Export FP16 ONNX, build a Triton model repository, call the HTTP endpoint | 45 min |

The fine-tune is intentionally short. It demonstrates the production workflow;
it is not a claim of improved domain accuracy.

## GPU support

The profile is selected from detected GPU memory when `LAB_PROFILE=auto`.

| GPU class | Profile | Lab 2 behavior |
|---|---|---|
| T4 16 GB | `t4` | Train CTC head only, 6-second clips |
| L4/A10 20-24 GB | `l4` | Train CTC head + last two encoder blocks, 10-second clips |
| A100 40/80 GB | `a100` | Full-model workshop path, 15-second clips |

Recommended Brev default: AWS `g6.2xlarge`, one L4, 32 GB host RAM, 150 GB
disk. This matches the AWS G6/L4 target in the event brief. A single A100 gives
more headroom; a T4 is the minimum supported GPU. Avoid ARM instances because
the workshop's pinned Python and Triton client wheels target x86_64.

At the time this package was prepared (6 August 2026), the Brev CLI showed
AWS `g6.2xlarge` at about US$1.17/hour and single A100 options from about
US$1.66/hour. Availability and pricing change, so check again before the event.

## Create the Brev Launchable

1. Push this directory to a public GitHub repository.
2. Follow [launchable/README.md](launchable/README.md).
3. Preview once on an L4, then run all four notebooks using the
   **Own Your Voice ASR Labs** kernel.
4. Share the generated Launchable link with attendees and ask them to deploy
   20-30 minutes before the lab starts.

The versioned builder settings are in
[`launchable/brev-launchable.yaml`](launchable/brev-launchable.yaml); the script
to paste into the Brev VM setup field is
[`launchable/setup.sh`](launchable/setup.sh).

## Run on an existing Brev instance

From this repository:

```bash
bash launchable/setup.sh
~/.venvs/own-your-voice-asr/bin/python scripts/preflight.py
```

Open Jupyter, select the **Own Your Voice ASR Labs** kernel and start with
`labs/00_start_here.ipynb`.

## Production handoff

The notebooks isolate four production contracts: 16 kHz audio preprocessing,
the model checkpoint, the ONNX model repository and the Triton inference API.
On AWS, the same artifacts can be put behind EKS storage, autoscaling,
observability and rollout policies. For a supported NVIDIA inference path,
Speech NIM packages optimized models with TensorRT and Triton and exposes
streaming/offline APIs; that extension requires the appropriate NGC access and
is deliberately not a workshop prerequisite.

See [references/SOURCES.md](references/SOURCES.md) for the official references.
