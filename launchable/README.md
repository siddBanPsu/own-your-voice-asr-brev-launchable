# Brev Console setup

This repository contains everything that can be versioned. The shareable
Launchable itself is created in the Brev Console.

1. Push this directory to a public GitHub repository.
2. In Brev, choose **Launchables → Create Launchable**.
3. Copy the name and description from `brev-launchable.yaml`.
4. Select **VM Mode**, enable managed Jupyter, and paste the full contents of
   `setup.sh` into the setup-script field.
5. For **Source**, select a public Git repository and enter its URL.
6. For **Network**, expose the managed Jupyter Secure Link on port 8888 and
   enable **Show as CTA**. On a fresh deployment, the setup script configures
   that CTA to land directly on `labs/00_start_here.ipynb`.
7. Set the default hardware to AWS `g6.2xlarge`, one L4 GPU, 32 GB RAM and
   150 GB storage. Participants may switch to a qualifying L4/A10 or a single
   A100 for the full path. A T4 is suitable only when skipping Lab 1.
8. Add the optional launch parameter `LAB_PROFILE`, default `auto`.
9. Keep access at **Anyone with the link** for a private roadshow, preview the
   deployment page, then create the Launchable.

The setup script installs managed CPython 3.12 environments with `uv`; it does
not use the Ubuntu image's system Python. Labs 0-2 use the isolated NeMo kernel,
and Lab 3 uses the isolated Riva client kernel recorded in its notebook
metadata. The split is required because NeMo 2.7.3 and Riva client 2.26.0 pin
incompatible Protobuf versions. Environment creation and the final CUDA
preflight fail if the interpreter is not Python 3.12. The selected Brev image
must also provide Docker with the NVIDIA Container Toolkit; Lab 0 reports
whether Docker is available. The script adds a user-level Jupyter Server
configuration that resolves the repository under `~/workspace` or directly
under the provider's home directory and redirects `/` to Lab 0. This applies
when managed Jupyter starts after setup; restart Jupyter once on an existing
instance that was already running.

The NeMo environment explicitly uses the official PyTorch 2.13.0 CUDA 12.6
wheel through `uv --torch-backend cu126`. Do not remove this selector: the
default PyPI wheel uses CUDA 13 and cannot initialize on the standard Brev
image when its driver reports CUDA 12.7 capability.

The setup script never asks for or persists a credential. Lab 1 separately
requires NVIDIA AI Enterprise entitlement and a personal NGC API key with
Catalog access to run the supported Parakeet CTC 0.6B Speech NIM. The notebook
requests the key with hidden input at runtime. Do not store it in the GitHub
repository, notebook, Launchable parameter default, or setup script.

Lab 3 asks for the same class of NGC credential at runtime to pull the pinned
Parakeet ASR NIM. The helper scripts use a temporary Docker configuration so
the key is not written into the attendee's long-lived Docker configuration.
Docker still prints an unencrypted-login warning for that temporary directory;
the scripts remove the directory automatically when each command finishes.

## Recommended rehearsal

Launch one L4 instance at least a day before the workshop and run
`labs/00_start_here.ipynb` followed by each lab's fast path. Confirm the NIM
entitlement and key before the rehearsal. Pre-cache the Dutch FLEURS splits and
confirm Lab 2 selects by `val_wer` and saves a complete `.nemo` checkpoint.
Lab 2 requests up to the same 400/50/100 duration-filtered records and uses the
same effective batch size and optimizer-step count on every supported GPU
profile. If fewer records meet the common duration limit, every profile uses
the same maximum available set; only the physical microbatch and trainable
encoder tail vary with GPU memory.
Model, dataset, and container downloads make the first run slower than later
runs. In Lab 3, confirm `nemo2riva` creates the `.riva` archive, `riva-build`
creates the RMIR, `riva-deploy` generates the target-GPU repository,
`custom_model.tar.gz` is served by the ASR NIM, and
the Riva gRPC client returns a transcript. If EKS is demonstrated, prepare that
cluster, S3 RMIR, and chart credentials separately.
