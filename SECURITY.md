# Security and Data Handling

Use only trusted local datasets, archives, and checkpoints. Research loaders
use ZIP extraction and pickle-capable `torch.load`/joblib operations; untrusted
inputs can be unsafe. These scripts are not a hardened upload-processing service.

Three model checkpoints, including learned normalization and PBS reference
statistics, are included. No credentials, raw images, or original dataset archives
are included. Keep new data and outputs outside version control. Review any outputs
before publication because filenames and exported metadata may identify samples.

The Flask deployment server is intended for a trusted local research network.
It does not provide authentication, TLS termination, or internet-facing service
hardening. Do not expose it directly to the public internet. Restrict network
access and review filesystem permissions on the Raspberry Pi.
