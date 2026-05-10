#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Server health check
if ! pgrep -x worldserver > /dev/null; then
    echo "ERROR: worldserver is not running" >&2
    exit 1
fi

if ! echo "PING" | nc -q 1 127.0.0.1 9000 | grep -q PONG 2>/dev/null; then
    echo "ERROR: NeuralBot TCP server not responding on port 9000" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/python/logs"
MODEL_DIR="$SCRIPT_DIR/python/checkpoints"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"
export NEURALBOT_MODEL="$MODEL_DIR/wow_neuralbot_${TIMESTAMP}"
export NEURALBOT_TIMESTEPS="${NEURALBOT_TIMESTEPS:-1000000}"

nohup python3 "$SCRIPT_DIR/python/train.py" > "$LOG_DIR/train_${TIMESTAMP}.log" 2>&1 &
echo "PID: $!"
echo "Log: $LOG_DIR/train_${TIMESTAMP}.log"
echo "Model: $NEURALBOT_MODEL.zip"
