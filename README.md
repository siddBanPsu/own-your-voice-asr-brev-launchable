# Own Your Voice: Parakeet ASR Labs on NVIDIA Brev

A GitHub-ready, single-GPU lab environment for the AWS + NVIDIA roadshow. The
sequence follows the event brief: deploy Parakeet CTC 0.6B through NVIDIA
Speech NIM, adapt the open model to domain speech, then export it to ONNX and
serve it with NVIDIA Triton Inference Server.

The repository is pinned to CPython 3.12. The Brev setup script installs its
own managed Python 3.12 runtime with `uv`, so it does not use Ubuntu 22.04's
older system Python.

## Lab flow

| Lab | Outcome | Workshop time |
|---|---|---:|
| 0. Start here | Verify CUDA, disk, Docker and the selected memory profile | 10 min |
| 1. NIM deployment | Deploy Parakeet CTC 0.6B Speech NIM and benchmark its HTTP API | 75-90 min |
| 2. Domain adaptation | Establish a held-out baseline, run a memory-aware micro-fine-tune, remeasure WER | 75 min |
| 3. ONNX + Triton | Export FP16 ONNX, build a Triton model repository, call the HTTP endpoint | 45 min |

The fine-tune is intentionally short. It demonstrates the production workflow;
it is not a claim of improved domain accuracy.

## GPU support

The profile is selected from detected GPU memory when `LAB_PROFILE=auto`.

| GPU class | Profile | Lab 2 behavior |
|---|---|---|
| T4 16 GB | `t4` | Labs 2-3 only; Speech NIM requires compute capability 8.0+ |
| L4/A10 20-24 GB | `l4` | Full path; NIM offline `bs=1`, then two-block adaptation |
| A100 40/80 GB | `a100` | Full path; NIM plus full-model workshop adaptation |

Recommended Brev default: AWS `g6.2xlarge`, one L4, 32 GB host RAM, 150 GB
disk. This matches the AWS G6/L4 target in the event brief. An L4 is the
minimum for the complete three-lab path; a single A100 gives more headroom.
Avoid ARM instances because the workshop's pinned Python and Triton client
wheels target x86_64.

For the complete three-lab path, use an L4, A10, A100, or another GPU listed in
the NVIDIA ASR NIM support matrix. A T4 can run the open-weight adaptation and
ONNX/Triton exercises but cannot run the Speech NIM deployment lab.

At the time this package was prepared (6 August 2026), the Brev CLI showed
AWS `g6.2xlarge` at about US$1.17/hour and single A100 options from about
US$1.66/hour. Availability and pricing change, so check again before the event.

## Speech NIM access

Lab 1 follows NVIDIA's supported Speech NIM deployment workflow for
`parakeet-0-6b-ctc-en-us`. Self-hosting requires NVIDIA AI Enterprise access
and a personal NGC API key with Catalog access. The notebook requests the key
with hidden input and passes it only to the NIM container for that runtime
session; never add it to the repository, notebook source, Launchable defaults,
or setup script. The Brev image must include Docker and the NVIDIA Container
Toolkit.

The default profile is the official low-memory single-client offline profile:
`name=parakeet-0-6b-ctc-en-us,bs=1,mode=ofl,diarizer=disabled,vad=default`.
It exposes HTTP on port 9000 and gRPC on 50051 only inside the Brev VM. Stop the
NIM with `bash scripts/stop_nim.sh` before Lab 2 to release GPU memory.

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
[`launchable/setup.sh`](launchable/setup.sh). That script provisions CPython
3.12 and fails the build if the resulting environment is not Python 3.12.

## Run on an existing Brev instance

From this repository:

```bash
bash launchable/setup.sh
~/.venvs/own-your-voice-asr/bin/python scripts/preflight.py
```

Open Jupyter, select the **Own Your Voice ASR Labs** kernel and start with
`labs/00_start_here.ipynb`.

## Production handoff

The notebooks isolate four production contracts: the Speech NIM API, 16 kHz
audio preprocessing, the open model checkpoint, and the ONNX/Triton model
repository. On AWS, deploy the supported NIM container through EKS with NGC
pull credentials, persistent model cache, health checks, metrics, GPU
scheduling, autoscaling, TLS, rollout policy, and workload-based load tests.
Labs 2-3 keep customization and the lower-level portable runtime boundary
visible rather than implying that a local export is a supported NIM package.

See [references/SOURCES.md](references/SOURCES.md) for the official references.
