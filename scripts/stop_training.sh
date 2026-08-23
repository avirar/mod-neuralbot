#!/usr/bin/env bash
# Safely stop the NeuralBot training stack.
# Usage: scripts/stop_training.sh [world]
#   (no args)   stop the trainer only (worldserver keeps running)
#   world       also stop the worldserver (clean SIGTERM shutdown)
#
# NOTE: do NOT put a literal "train_v3.py" anywhere else in a command that runs
# this script's pkill — the [t] bracket trick only protects the pkill's own
# pattern, not other occurrences in the same shell.
set -u
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODULE_DIR"

touch .maintenance
echo "maintenance flag set (watchdog standing down)"

if pkill -f "[t]rain_v3.py"; then
    echo "trainer stopped"
else
    echo "trainer not running"
fi

if [ "${1:-}" = "world" ]; then
    if pkill -x worldserver; then
        echo "worldserver stopping (clean shutdown, ~20s)"
    else
        echo "worldserver not running"
    fi
fi
