#!/usr/bin/env bash
# Run random cluster baseline (no feature-based clustering).
set -euo pipefail

cd "$(dirname "$0")/.."

python3 discover.py --config config/random.yaml "$@"
