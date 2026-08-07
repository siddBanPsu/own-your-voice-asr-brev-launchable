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
   150 GB storage. Participants may switch to a single A100 or a qualifying
   T4/L4/A10 before deployment.
8. Add the optional launch parameter `LAB_PROFILE`, default `auto`.
9. Keep access at **Anyone with the link** for a private roadshow, preview the
   deployment page, then create the Launchable.

The setup script never asks for or persists a credential. Parakeet CTC 0.6B is
an open model. If your organization requires authenticated Hugging Face access,
use a runtime secret manager rather than adding a token to this repository.

## Recommended rehearsal

Launch one L4 instance at least a day before the workshop and run
`labs/00_start_here.ipynb` followed by each lab's fast path. Model and container
downloads make the first run slower than subsequent runs.
