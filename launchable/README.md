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
   enable **Show as CTA**.
7. Set the default hardware to AWS `g6.2xlarge`, one L4 GPU, 32 GB RAM and
   150 GB storage. Participants may switch to a qualifying L4/A10 or a single
   A100 for the full path. A T4 is suitable only when skipping Lab 1.
8. Add the optional launch parameter `LAB_PROFILE`, default `auto`.
9. Keep access at **Anyone with the link** for a private roadshow, preview the
   deployment page, then create the Launchable.

The setup script installs a managed CPython 3.12 runtime with `uv`; it does not
use the Ubuntu image's system Python. Both environment creation and the final
CUDA preflight fail if the interpreter is not Python 3.12. The selected Brev
image must also provide Docker with the NVIDIA Container Toolkit; Lab 0 reports
whether Docker is available.

The setup script never asks for or persists a credential. Lab 1 separately
requires NVIDIA AI Enterprise entitlement and a personal NGC API key with
Catalog access to run the supported Parakeet CTC 0.6B Speech NIM. The notebook
requests the key with hidden input at runtime. Do not store it in the GitHub
repository, notebook, Launchable parameter default, or setup script.

## Recommended rehearsal

Launch one L4 instance at least a day before the workshop and run
`labs/00_start_here.ipynb` followed by each lab's fast path. Confirm the NIM
entitlement and key before the rehearsal. Model and container downloads make
the first run slower than subsequent runs.
