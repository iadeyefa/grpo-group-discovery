# Implementation guidelines

- Keep clustering **offline** — no training-loop or GPU dependencies.
- Match artifact formats documented in `docs/artifact-format.md`.
- Add debug logs (`logging.debug`) at pipeline stage boundaries.
- Prefer small, focused modules over monolithic scripts.
- Version outputs under `outputs/<run_name>/` with reproducible metadata (seed, K, features).
