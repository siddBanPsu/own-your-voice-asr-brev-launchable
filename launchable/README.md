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
   that CTA to land directly on `labs/00_start_here.ipynb`. Add a second Secure
   Link named `tensorboard` on port 6006, but leave **Show as CTA** disabled;
   it remains dormant unless the optional Lab 2 logger and TensorBoard process
   are enabled.
7. Set the default hardware to AWS `g6.2xlarge`, one L4 GPU, 32 GB RAM and
   150 GB storage. Participants may switch to a qualifying L4/A10, a single
   A100, or a single RTX PRO 6000 Blackwell for the full path. A T4 is suitable
   only when skipping Lab 1.
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

The NeMo environment explicitly selects the official PyTorch 2.13.0 CUDA wheel:
`cu126` for the L4/A100 path, and `cu129` for RTX PRO 6000 Blackwell (`sm_120`).
Do not remove this selector: the default CUDA 13 PyPI wheel can require a newer
driver than the standard Brev image exposes.

TensorBoard 2.20.0 and its compatible Setuptools 80.9.0 runtime are installed
only in the NeMo environment. Lab 2 keeps `ENABLE_TENSORBOARD = False` by
default. When enabled, start the dashboard from a Brev terminal using the
command in the main README, then open the port 6006 Secure Link. Do not add
port 6006 to the public TCP/UDP list.

The main README deliberately changes into the repository and passes
`"${PWD}/artifacts/tensorboard"` to TensorBoard. Do not shorten this to a
relative `artifacts/tensorboard` path: Brev terminals can open in the user's
home directory while Lab 2 writes beneath the repository. An existing
Launchable provisioned with the older setup must receive the documented
one-time Setuptools 80.9.0 repair. Pulling the updated repository does not
rebuild its Python environment, and the VM setup script does not automatically
run again on restart. A fresh deployment with this setup script applies and
verifies the compatible pin automatically.

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

If the event uses an RTX PRO 6000 Blackwell instead, repeat this complete
rehearsal on that GPU after preflight confirms `sm_120` support. The Python
wheel selection is automated, but NIM/TensorRT and Riva engine generation must
be validated on the target GPU; do not reuse a generated Riva repository from
another GPU product.

For an ONNX inspection exercise, set `SAVE_INTERMEDIATE_ONNX = True` in Lab 3.
The extra NeMo export is stored at
`artifacts/onnx/parakeet-ctc-0.6b-nl.onnx`; it is optional and does not replace
the `.riva` archive used to build the RMIR.
