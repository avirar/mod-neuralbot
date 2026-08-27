#!/usr/bin/env bash
# NeuralBot watchdog: keeps ALL training instances alive unattended (multi-day runs).
# Per instance: worldserver dead -> restart it; trainer dead >5 min -> restart it,
# resuming that instance's newest checkpoint. Maintenance flag -> do nothing.
# Run under the systemd user service neuralbot-watchdog.service (Restart=always).
#
# Instance layout (see AGENTS.md "Multi-instance scaling"):
#   1: default conf,  shm /neuralbot_shm,  nbot/Neuralbot, acore_characters,  model v15_dense (slow decay 0.25)
#   2: instance2 conf, shm /neuralbot_shm2, xbot/Xbot,      acore_characters2, model i2       (slow decay 0.25)
#   3: instance3 conf, shm /neuralbot_shm3, ybot/Ybot,      acore_characters3, model i3       (FAST decay 0.05)
set -u

MODULE_DIR="/home/luke/GIT/azerothcore-wotlk/modules/mod-neuralbot"
BIN_DIR="/home/luke/GIT/azerothcore-wotlk/env/dist/bin"
FLAG="$MODULE_DIR/.maintenance"
LOG="$MODULE_DIR/python/logs/watchdog.log"
VENV="$MODULE_DIR/.venv/bin/activate"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# name|conf-arg|shm|prefix|chardb|model|decay_frac
INSTANCES=(
    "1||/neuralbot_shm|nbot|acore_characters|wow_neuralbot_model_v15_dense|0.05"
    "2|../etc/instance2/worldserver.conf|/neuralbot_shm2|xbot|acore_characters2|wow_neuralbot_model_i2|0.25"
    "3|../etc/instance3/worldserver.conf|/neuralbot_shm3|ybot|acore_characters3|wow_neuralbot_model_i3|0.05"
)

online_bots() { # $1=prefix $2=chardb
    mysql -u acore -pabc -h 127.0.0.1 -N -e "SELECT COUNT(*) FROM $2.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^$1[0-9]+$' AND c.online=1;" 2>/dev/null
}

worldserver_pid() { # $1=conf-arg (empty = instance 1, no -c)
    if [ -n "$1" ]; then pgrep -f "worldserver -c $1" | head -1
    else pgrep -x worldserver | while read -r p; do grep -aq "instance" /proc/$p/cmdline 2>/dev/null || { echo "$p"; break; }; done
    fi
}

trainer_pid_for_model() { # $1=model
    for p in $(pgrep -f "[t]rain_v3.py"); do
        if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^NEURALBOT_MODEL=$1$"; then echo "$p"; return; fi
    done
}

start_worldserver() { # $1=conf-arg $2=prefix $3=shmname-env
    log "instance $2: starting worldserver"
    cd "$BIN_DIR"
    if [ -n "$1" ]; then
        # shellcheck disable=SC2086
        $3 nohup ./worldserver -c "$1" > "ws_$2.out" 2>&1 &
    else
        setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver' >> ws_console.out 2>&1 < /dev/null &
    fi
    # Yield to interactive use (games/desktop) under CPU contention only.
    renice -n 5 -p $! >/dev/null 2>&1 || true
}

start_trainer() { # $1=model $2=shm-path $3=decay_frac $4=chardb
    local newest
    newest=$(ls -1t "$MODULE_DIR"/$1*_steps.zip 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        cp "$newest" "$MODULE_DIR/$1.zip"
        log "resuming $1 from $newest"
    else
        rm -f "$MODULE_DIR/$1.zip"
        log "starting fresh trainer $1 (no checkpoint)"
    fi
    cd "$MODULE_DIR"
    source "$VENV"
    # Hyperparameters track the active experiments (KL guard off: see train_v3.py).
    NEURALBOT_MODEL="$1" NEURALBOT_TIMESTEPS=1000000000 \
        NEURALBOT_LR="${NEURALBOT_LR:-1.5e-4}" NEURALBOT_ENT="${NEURALBOT_ENT:-0.01}" \
        NEURALBOT_ENT_FINAL="${NEURALBOT_ENT_FINAL:-0.0005}" NEURALBOT_ENT_DECAY_FRAC="$3" \
        NEURALBOT_KL="${NEURALBOT_KL:-0}" NEURALBOT_NUM_BOTS="${NEURALBOT_NUM_BOTS:-800}" \
        NEURALBOT_REWARD_MODE="${NEURALBOT_REWARD_MODE:-symlog}" \
        NEURALBOT_BATCH_SIZE=4096 NEURALBOT_EPOCHS=5 NEURALBOT_NSTEPS=1024 \
        SHM_PATH="$2" NEURALBOT_DB_NAME="$4" \
        nohup python3 -u python/train_v3.py > "python/logs/train_auto_$1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    disown
    # Trainers yield to interactive use (games/desktop) under CPU contention only.
    renice -n 5 -p $! >/dev/null 2>&1 || true
    log "$1 trainer launched (pid $!)"
}

mkdir -p "$MODULE_DIR/python/logs"
touch "$LOG"

if [ -e "$FLAG" ]; then
    log "maintenance flag present — standing down"
    exit 0
fi

for spec in "${INSTANCES[@]}"; do
    IFS='|' read -r ID CONF SHM PREFIX CHARDB MODEL DECAY <<< "$spec"

    if [ -z "$(worldserver_pid "$CONF")" ]; then
        log "instance $ID: worldserver DOWN"
        case "$ID" in
            2) ENVV='AC_NEURAL_BOT_SHM_NAME=/neuralbot_shm2 AC_NEURAL_BOT_BOT_CHARACTER_NAME=Xbot AC_NEURAL_BOT_BOT_ACCOUNT_PREFIX=xbot' ;;
            3) ENVV='AC_NEURAL_BOT_SHM_NAME=/neuralbot_shm3 AC_NEURAL_BOT_BOT_CHARACTER_NAME=Ybot AC_NEURAL_BOT_BOT_ACCOUNT_PREFIX=ybot' ;;
            *) ENVV='' ;;
        esac
        start_worldserver "$CONF" "$ID" "$ENVV"
    fi

    if [ -z "$(trainer_pid_for_model "$MODEL")" ]; then
        GRACE="/tmp/neuralbot_trainer_died_$MODEL"
        if [ ! -e "$GRACE" ]; then
            touch "$GRACE"
            log "instance $ID: trainer $MODEL not running — grace period starts"
            continue
        fi
        AGE=$(( $(date +%s) - $(stat -c %Y "$GRACE") ))
        if [ "$AGE" -ge 300 ]; then
            rm -f "$GRACE"
            N=$(online_bots "$PREFIX" "$CHARDB")
            if [ "${N:-0}" -ge 790 ]; then
                start_trainer "$MODEL" "/dev/shm$SHM" "$DECAY" "$CHARDB"
            else
                log "instance $ID: bots=${N:-0} — worldserver unhealthy, not starting trainer"
            fi
        fi
    else
        rm -f "/tmp/neuralbot_trainer_died_$MODEL"
    fi
done
log "ok: $(pgrep -c -x worldserver) worldservers, $(pgrep -f '[t]rain_v3.py' | wc -l) trainers detected"
