#!/usr/bin/env bash
# Start the NeuralBot training stack (worldserver + trainer). The manual
# counterpart to scripts/stop_training.sh. Safe to re-run — skips whatever is
# already up, and clears the .maintenance flag so the watchdog resumes management.
# Usage: scripts/start_training.sh [model_prefix] [timesteps]
set -u
MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="/home/luke/GIT/azerothcore-wotlk/env/dist/bin"
MODEL_PREFIX="${1:-wow_neuralbot_model_v13}"
STEPS="${2:-1000000000}"
DB="mysql -u acore -pabc -h 127.0.0.1 -N -e"
LOG="$MODULE_DIR/python/logs/watchdog.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; echo "$*"; }

online_bots() {
    $DB "SELECT COUNT(*) FROM acore_characters.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^nbot[0-9]+$' AND c.online=1;" 2>/dev/null
}

trainer_alive() { pgrep -f "[t]rain_v3.py" > /dev/null; }

# ── 1. worldserver ─────────────────────────────────────────────────────────
if pgrep -x worldserver > /dev/null; then
    log "worldserver already running"
else
    log "starting worldserver"
    cd "$BIN_DIR"
    setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver' >> ws_console.out 2>&1 < /dev/null &
    for i in $(seq 1 60); do
        sleep 5
        N=$(online_bots)
        if [ "${N:-0}" = "400" ]; then log "worldserver up, 400 bots (${i}x5s)"; break; fi
        pgrep -x worldserver > /dev/null || { log "worldserver exited during startup — check ws_console.out"; exit 1; }
    done
    if [ "${N:-0}" != "400" ]; then
        log "worldserver up but bots=${N:-0} after 300s — starting trainer anyway"
    fi
fi

# ── 2. trainer ─────────────────────────────────────────────────────────────
if trainer_alive; then
    log "trainer already running"
else
    newest=$(ls -1t "$MODULE_DIR"/${MODEL_PREFIX}*_steps.zip 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        cp "$newest" "$MODULE_DIR/${MODEL_PREFIX}.zip"
        log "resuming trainer from $newest"
    else
        rm -f "$MODULE_DIR/${MODEL_PREFIX}.zip"
        log "starting fresh trainer (no checkpoint)"
    fi
    cd "$MODULE_DIR"
    source .venv/bin/activate
    NEURALBOT_MODEL="$MODEL_PREFIX" NEURALBOT_TIMESTEPS="$STEPS" \
        NEURALBOT_LR="${NEURALBOT_LR:-1.5e-4}" NEURALBOT_ENT="${NEURALBOT_ENT:-0.01}" NEURALBOT_KL="${NEURALBOT_KL:-0}" \
        NEURALBOT_REWARD_MODE="${NEURALBOT_REWARD_MODE:-symlog}" \
        nohup python3 -u python/train_v3.py > "python/logs/train_manual_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    disown
    log "trainer launched (pid $!)"
fi

# ── 3. re-enable the watchdog ──────────────────────────────────────────────
rm -f "$MODULE_DIR/.maintenance"
log "maintenance flag cleared — watchdog active"
