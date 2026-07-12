#!/usr/bin/env bash
# Run floor baseline: preference-similarity discovery.
set -euo pipefail

cd "$(dirname "$0")"/..

python3 discover.py --config config/preference_similarity.yaml "$@"
