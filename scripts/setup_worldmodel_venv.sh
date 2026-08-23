#!/usr/bin/env bash
# Create the world-model spike venv: Python 3.11 + torch cu128 (Blackwell sm_120).
# See WORLD_MODEL_SPIKE.md. Run in the background — torch is ~3 GB, ~10 min.
# Usage: scripts/setup_worldmodel_venv.sh
set -euo pipefail
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$MODULE_DIR/worldmodel/.venv311"
mkdir -p "$MODULE_DIR/worldmodel"

if [ -x "$VENV/bin/python" ]; then
    echo "venv already exists at $VENV"
else
    echo "creating Python 3.11 venv via uv..."
    uv venv --python 3.11 "$VENV"
fi

echo "installing torch==2.8.0+cu128..."
uv pip install --python "$VENV/bin/python" torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "smoke test:"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
echo "done"
