# Own Your Voice: Parakeet, NeMo, and Riva on NVIDIA Brev

A single-GPU workshop that connects NVIDIA's speech stack end to end:

```text
Parakeet Speech NIM baseline
        ↓
NeMo fine-tuning + held-out evaluation
        ↓
selected .nemo checkpoint
        ├── optional standalone ONNX copy
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
| T4 16 GB | `t4` | Labs 2-3 only; train the CTC decoder with smaller physical microbatches |
| L4/A10 20-24 GB | `l4` | Train the decoder plus the final two encoder blocks |
| A100 40/80 GB | `a100` | Same safe default, with room to increase the controls |

Lab 2 keeps the comparison workload fixed across profiles: up to 400 train, 50
validation, and 100 held-out test utterances; a common six-second duration
limit; effective batch size four; and 400 optimizer steps. If a split contains
fewer eligible utterances, the notebook warns, uses every available record, and
reports both requested and actual counts. Physical batch size and gradient
accumulation adapt to GPU memory, while the trainable encoder tail remains
profile-specific. As a result, all profiles evaluate the same records and see
the same total sample exposure, although CUDA kernels can still prevent
bit-for-bit identical results across GPU architectures.

Recommended Brev default: AWS `g6.2xlarge`, one L4, 32 GB host RAM, and at
least 150 GB disk. Recheck provider availability, price, image, and capacity
before the event. Stop the Lab 1 NIM before training or Riva deployment so only
one speech stack owns the GPU.

## Optional TensorBoard training dashboard

Lab 2 defines `ENABLE_TENSORBOARD = False`, so the default workshop run does
not create TensorBoard event files. Set the flag to `True` before executing the
training cell to record versioned runs under
`artifacts/tensorboard/parakeet-nl/`. Each rerun receives a separate
`version_N` directory.

Start the dashboard from the repository root so TensorBoard receives the
absolute directory that Lab 2 writes to. A standard Brev source checkout uses
the first command below; if Jupyter shows the repository under `~/workspace`,
change into that checkout instead.

```bash
cd ~/own-your-voice-asr-brev-launchable
~/.venvs/own-your-voice-asr/bin/tensorboard \
  --logdir "${PWD}/artifacts/tensorboard" \
  --host 0.0.0.0 \
  --port 6006
```

Keep that terminal running while using the dashboard. To confirm that Lab 2
has actually written event files, run:

```bash
find "${PWD}/artifacts/tensorboard" \
  -type f -name 'events.out.tfevents.*' -print
```

### Existing Launchables and `pkg_resources`

If the Launchable was provisioned before the TensorBoard dependency fix, it
retains the virtual environment created by the older setup script. A `git pull`
updates repository files but does not rebuild that environment, and Brev does
not rerun the VM setup script on restart. If TensorBoard reports
`No module named 'pkg_resources'`, repair that existing environment once:

```bash
~/.local/bin/uv pip install \
  --python ~/.venvs/own-your-voice-asr/bin/python \
  setuptools==80.9.0
```

The older setup installed the newest available Setuptools. Setuptools removed
`pkg_resources` in version 82.0.0, but TensorBoard 2.20 still imports it. Fresh
deployments using the updated setup pin Setuptools 80.9.0 in both installation
phases and fail setup verification if either that version or `pkg_resources`
is missing. A deprecation warning about `pkg_resources` is expected and is not
a startup failure.

In the Brev Launchable Network settings, add a Secure Link named
`tensorboard` for port 6006 and leave **Show as CTA** disabled. Do not expose
6006 as a public TCP port. Open that Secure Link after the command reports that
TensorBoard is serving. The dashboard can show the metrics emitted by the NeMo
Lightning module, including training loss, learning rate, and validation WER.
It cannot reconstruct runs completed while the flag was `False`. If the
event-file check above prints nothing, set `ENABLE_TENSORBOARD = True`, rerun
the experiment-controls cell, and rerun the training cell.

## Access requirements

Lab 1 and Lab 3 require NVIDIA NGC access and the applicable NVIDIA AI
Enterprise entitlement for the Speech NIM containers.

### Get an NGC personal key before launch

1. Sign in to the [NGC API Keys page](https://org.ngc.nvidia.com/setup/api-keys).
2. Select **Generate Personal Key**, give it a descriptive workshop name, and
   choose an appropriate expiration date.
3. Include at least **NGC Catalog** under **Services Included**, as required by
   the [Speech NIM access guide](https://docs.nvidia.com/nim/speech/latest/get-started/ngc-access-setup.html).
4. Generate the key, copy it immediately, and store it in a password manager or
   another secure location. NGC does not retain the complete key for later
   display.

A personal key authenticates the NGC account; it does not grant Speech NIM or
NVIDIA AI Enterprise entitlement. Confirm the account can access the required
containers before the workshop. The [NGC key documentation](https://docs.nvidia.com/ngc/latest/ngc-private-registry-user-guide.html#ngc-api-keys)
also covers rotation, expiration, and registry authentication.

Labs 1 and 3 ask for the key with hidden input. The scripts use a temporary
Docker configuration and do not add the key to the repository, notebook,
Launchable defaults, or long-lived Docker configuration.

Lab 2 uses the open Parakeet checkpoint with NVIDIA NeMo 2.7.3. Lab 3 uses
`nemo2riva` 2.22.0 to export the `.nemo` checkpoint, the published Parakeet 0.6B
ASR NIM 3.1.0 for ServiceMaker and serving, the Riva NIM Helm chart 1.1.0 on
EKS, and the isolated Riva Python client 2.26.0.

## Lab 3 deployment choices

### Amazon EKS — production-oriented path

The RMIR from Lab 3 is staged in S3 and selected by the Speech NIM Helm chart.
The chart runs target-GPU optimization, generates the Triton model repository,
starts the Riva API, and exposes a Kubernetes Service. See
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

The first script exports `.nemo → .riva` and calls the container's positional
`riva-build speech_recognition` interface. `riva-deploy` then optimizes the RMIR
for the current GPU, packages
`custom_model.tar.gz`, and the ASR NIM serves it on HTTP port 9000 and Riva
gRPC port 50051. Applications call Riva, not raw Triton. Stop it with:

```bash
bash scripts/stop_riva.sh
```

Lab 3 also has an opt-in `SAVE_INTERMEDIATE_ONNX` control. Set it to `True` to
retain `artifacts/onnx/parakeet-ctc-0.6b-nl.onnx` from the selected `.nemo`
checkpoint. This is a standalone teaching and inspection artifact, adds another
export pass, and is not consumed by the `.riva → RMIR` deployment path.

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

The NeMo environment installs the official PyTorch 2.13.0 CUDA 12.6 wheel
explicitly (`uv --torch-backend cu126`). This remains compatible with the CUDA
12.7-capable driver in the standard Brev image instead of allowing PyPI to
select the default CUDA 13 build.

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
version, Parakeet checkpoint, dataset size, ASR NIM image, target GPU, RMIR
build time, engine-generation time, Riva transcript, and EKS chart/storage
configuration. Do not describe static checks as a live Riva or EKS deployment.

See [`references/SOURCES.md`](references/SOURCES.md) for official references.
