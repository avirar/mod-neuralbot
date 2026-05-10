#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Server health check
if ! pgrep -x worldserver > /dev/null; then
    echo "ERROR: worldserver is not running" >&2
    exit 1
fi

# Wait for bots to be logged in
echo "Waiting for bots to log in..."
for i in $(seq 1 60); do
    COUNT=$(echo "BOTS" | nc -q 2 -w 2 127.0.0.1 9000 2>/dev/null | wc -w)
    if [ "$COUNT" -ge 20 ]; then
        echo "All $((COUNT - 1)) bots ready after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Only $((COUNT - 1)) bots ready after 60s" >&2
        exit 1
    fi
    sleep 1
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/python/logs"
MODEL_DIR="$SCRIPT_DIR/python/checkpoints"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"
export NEURALBOT_MODEL="$MODEL_DIR/wow_neuralbot_${TIMESTAMP}"
export NEURALBOT_TIMESTEPS="${NEURALBOT_TIMESTEPS:-5000000}"

nohup python3 "$SCRIPT_DIR/python/train.py" > "$LOG_DIR/train_${TIMESTAMP}.log" 2>&1 &
echo "PID: $!"
echo "Log: $LOG_DIR/train_${TIMESTAMP}.log"
echo "Model: $NEURALBOT_MODEL.zip"
