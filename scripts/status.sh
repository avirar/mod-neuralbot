#!/usr/bin/env bash
# One-shot health snapshot for the NeuralBot stack (training + BC recorder).
# Usage: scripts/status.sh
set -u
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODULE_DIR"
DB="mysql -u acore -pabc -h 127.0.0.1 -N -e"

echo "=== processes ==="
pgrep -x worldserver > /dev/null && echo "worldserver: UP" || echo "worldserver: DOWN"
pgrep -f "python3 python/train_v3" > /dev/null && echo "trainer:     UP" || echo "trainer:     DOWN"

echo
echo "=== bots online ==="
$DB "SELECT CONCAT('neuralbots: ', COUNT(*)) FROM acore_characters.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^nbot[0-9]+$' AND c.online=1 UNION ALL SELECT CONCAT('randombots: ', COUNT(*)) FROM acore_characters.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^rndbot' AND c.online=1;" 2>/dev/null

echo
echo "=== BC recorder ==="
DEMO="python/bc_demos/demos.bin"
if [ -f "$DEMO" ]; then
    SIZE=$(stat -c%s "$DEMO")
    echo "demos.bin: $((SIZE / 1048576)) MB (~$((SIZE / 5997)) records)"
else
    echo "demos.bin: (none)"
fi

echo
echo "=== latest trainer log ==="
LOG=$(ls -t python/logs/train_v13_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo "file: $LOG"
    grep -E "fps|entropy_loss|explained_variance|approx_kl|value_loss|total_timesteps" "$LOG" | tail -10
else
    echo "(no v13 log)"
fi

echo
echo "=== disk ==="
du -sh . 2>/dev/null
df -h "$MODULE_DIR" 2>/dev/null | tail -1
