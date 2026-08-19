#!/usr/bin/env bash
# NeuralBot watchdog: keeps training alive unattended (multi-day runs).
# - worldserver dead            -> restart it, wait for 400 bots
# - trainer dead >5 min         -> restart it, resuming newest v5 checkpoint
# - maintenance flag set        -> do nothing (agent/user is working on the stack)
# Run under the systemd user service neuralbot-watchdog.service (Restart=always).
set -u

MODULE_DIR="/home/luke/GIT/azerothcore-wotlk/modules/mod-neuralbot"
BIN_DIR="/home/luke/GIT/azerothcore-wotlk/env/dist/bin"
FLAG="$MODULE_DIR/.maintenance"
MODEL_PREFIX="wow_neuralbot_model_v12"
LOG="$MODULE_DIR/python/logs/watchdog.log"
DB="mysql -u acore -pabc -h 127.0.0.1 -N -e"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

online_bots() {
    $DB "SELECT COUNT(*) FROM acore_characters.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^nbot[0-9]+$' AND c.online=1;" 2>/dev/null
}

trainer_alive() { pgrep -f "[t]rain_v3.py" > /dev/null; }

start_worldserver() {
    log "starting worldserver"
    cd "$BIN_DIR"
    setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver' >> ws_console.out 2>&1 < /dev/null &
    for i in $(seq 1 60); do
        sleep 5
        N=$(online_bots)
        if [ "${N:-0}" = "400" ]; then log "worldserver up, 400 bots (${i}x5s)"; return 0; fi
        pgrep -x worldserver > /dev/null || { log "worldserver exited during startup"; return 1; }
    done
    log "worldserver up but bots=${N:-0} after 300s"
    return 0
}

start_trainer() {
    # Resume from the newest v5 checkpoint if one exists
    local newest
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
    NEURALBOT_MODEL="$MODEL_PREFIX" NEURALBOT_TIMESTEPS=1000000000 NEURALBOT_LR="${NEURALBOT_LR:-2.5e-4}" \
        nohup python3 python/train_v3.py > "python/logs/train_auto_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    disown
    log "trainer launched (pid $!)"
}

mkdir -p "$MODULE_DIR/python/logs"
touch "$LOG"

if [ -e "$FLAG" ]; then
    log "maintenance flag present — standing down"
    exit 0
fi

# worldserver health
if ! pgrep -x worldserver > /dev/null; then
    log "worldserver DOWN"
    start_worldserver
    sleep 10
fi

# trainer health (5-minute grace via flag file timestamp)
if ! trainer_alive; then
    if [ ! -e /tmp/neuralbot_trainer_died ]; then
        touch /tmp/neuralbot_trainer_died
        log "trainer not running — grace period starts"
        exit 0
    fi
    AGE=$(( $(date +%s) - $(stat -c %Y /tmp/neuralbot_trainer_died) ))
    if [ "$AGE" -ge 300 ]; then
        rm -f /tmp/neuralbot_trainer_died
        N=$(online_bots)
        if [ "${N:-0}" -ge 390 ]; then
            start_trainer
        else
            log "bots=${N:-0} — worldserver unhealthy, not starting trainer"
        fi
    fi
else
    rm -f /tmp/neuralbot_trainer_died
    log "ok: worldserver up, trainer running (pid $(pgrep -f '[t]rain_v3.py' | head -1))"
fi
