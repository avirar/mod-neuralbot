#!/usr/bin/env bash
# Run the BC demonstration analyzer on the recorded playerbot actions.
# Usage: scripts/bc_report.sh [hist|progress] [--limit N]
set -euo pipefail
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODULE_DIR"
source .venv/bin/activate
CMD="${1:-hist}"
shift 2>/dev/null || true
python3 python/bc_analyze.py "$CMD" python/bc_demos/demos.bin "$@"
