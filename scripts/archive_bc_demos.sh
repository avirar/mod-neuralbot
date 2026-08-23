#!/usr/bin/env bash
# Archive the BC demonstration recording to the NAS.
# The recorder opens demos.bin in append mode and keeps the handle, so we move
# (not truncate) the file and create a fresh empty one — but only when the
# recorder is stopped, to avoid splitting the stream mid-write.
# Usage: scripts/archive_bc_demos.sh [--force]
set -euo pipefail
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODULE_DIR"
NAS_DIR="${NAS_DIR:-$HOME/NAS/temp/neuralbot}"
DEMO="python/bc_demos/demos.bin"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$NAS_DIR/bc_demos"

[ -f "$DEMO" ] || { echo "no demos.bin"; exit 0; }

# Refuse while the recorder is actively writing, unless --force.
if [ "${1:-}" != "--force" ]; then
    if pgrep -f "python3 python/train_v3" > /dev/null && grep -q "BcRecordPath" /home/luke/GIT/azerothcore-wotlk/env/dist/etc/modules/mod-neuralbot.conf; then
        # Check whether BcRecordPath is actually enabled (non-empty path).
        if grep -qE '^NeuralBot.BcRecordPath = ".+"' /home/luke/GIT/azerothcore-wotlk/env/dist/etc/modules/mod-neuralbot.conf; then
            echo "recorder appears enabled — stop training first (scripts/stop_training.sh world) or pass --force"
            exit 1
        fi
    fi
fi

SIZE=$(stat -c%s "$DEMO")
gzip -c "$DEMO" > "$NAS_DIR/bc_demos/demos_${STAMP}.bin.gz"
rm -f "$DEMO"
touch "$DEMO"
echo "archived $((SIZE / 1048576)) MB (~$((SIZE / 5997)) records) -> $NAS_DIR/bc_demos/demos_${STAMP}.bin.gz"
