# Security

Please report vulnerabilities privately to nsastrophysics@gmail.com rather than in a public
issue. I aim to reply within a week.

What the package does to stay safe:

- There is no network access at run time. The only network code is `declinecast fetch`, which
  you run yourself; it requests the fourteen published ECMC file URLs one at a time, follows no
  links, keeps TLS verification on, and records the SHA-256 of every file it saves.
- The bundled extract's SHA-256 is checked every time it is loaded.
- Saved models: the SHA-256 of every file is recorded in `manifest.json` and checked on load.
  Weights are a PyTorch `state_dict` read with `torch.load(weights_only=True)`, which refuses
  anything but tensors and plain containers; the network is rebuilt from the settings in the
  manifest, not unpickled. Loading refuses any file the manifest does not list. No pickle
  files are written or read. Only load model directories from sources you trust.
- SQL runs in an in-memory DuckDB database inside the Python process. Queries after the panel
  build see only the tables the package registers: external access (files, network) and
  extension auto-loading are switched off. The panel build may read the raw files you point it
  at, but loads no extensions, so it has no network access either.
- Treat downloaded raw files as untrusted data: keep them in their own folder, and run Python
  that reads them with `python -I`.
- Output is written only to the directory you name, and existing files are not overwritten
  without `--overwrite`.
- Releases are published from GitHub Actions with PyPI Trusted Publishing, so no tokens are
  stored.
