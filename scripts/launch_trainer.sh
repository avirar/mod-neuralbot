#!/usr/bin/env bash
# Launch the NeuralBot trainer detached (avoids pkill self-match traps).
# Usage: launch_trainer.sh <model_prefix> [timesteps]
set -euo pipefail
MODEL="${1:?model prefix required}"
STEPS="${2:-1000000000}"
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODULE_DIR"
source .venv/bin/activate
NEURALBOT_MODEL="$MODEL" NEURALBOT_TIMESTEPS="$STEPS" \
    nohup python3 python/train_v3.py > "python/logs/train_${MODEL##*model_}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
disown
echo "launched $MODEL ($STEPS steps), pid $!"
