#!/usr/bin/env bash
# Run preference-similarity clustering with default config.
set -euo pipefail

cd "$(dirname "$0")/.."

python discover.py --config config/default.yaml "$@"
