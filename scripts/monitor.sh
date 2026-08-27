#!/usr/bin/env bash
# NeuralBot long-horizon monitor: one sample of all 3 arms + event detection.
# Appends a compact line to python/logs/monitor.log; prints events to stdout.
#
# Arms:
#   1 v15_dense (long lineage, slow decay 0.25)   -> acore_characters
#   2 i2        (fresh baseline, slow decay 0.25) -> acore_characters2
#   3 i3        (warm from i2, FAST decay 0.05)   -> acore_characters3
#
# Phase-0 verdict signal: policy entropy H = -entropy_loss/ent_coef for arm 2 vs 3.
#   i3's H collapses faster than i2's while its reward/kill improves => entropy was binding.
#   i2/i3 track identically => entropy NOT binding => go to RL_RESEARCH.md §10 Phase 1.
set -u
MODULE_DIR="/home/luke/GIT/azerothcore-wotlk/modules/mod-neuralbot"
cd "$MODULE_DIR"
LOG="$MODULE_DIR/python/logs/monitor.log"
DB="mysql -u acore -pabc -h 127.0.0.1 -N -e"
TS=$(date '+%F %T')

spec=(
  "1|wow_neuralbot_model_v15_dense|acore_characters|nbot"
  "2|wow_neuralbot_model_i2|acore_characters2|xbot"
  "3|wow_neuralbot_model_i3|acore_characters3|ybot"
)

trainer_pid() { # $1=model
    for p in $(pgrep -f "[t]rain_v3.py"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^NEURALBOT_MODEL=$1$" && { echo "$p"; return; }
    done
}

# count worldservers for an instance by classifying /proc/<pid>/cmdline
ws_count() { # $1=instance id
    local n=0 pid
    for pid in $(pgrep -x worldserver); do
        local c; c=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
        case "$1" in
            1) echo "$c" | grep -qv "instance" && n=$((n+1)) ;;
            2) echo "$c" | grep -q "instance2" && n=$((n+1)) ;;
            3) echo "$c" | grep -q "instance3" && n=$((n+1)) ;;
        esac
    done
    echo "$n"
}

# newest log whose content mentions the model name
latest_log() { # $1=model
    find "$MODULE_DIR"/python/logs -name 'train_*.log' -newermt '-24 hours' -print0 2>/dev/null \
        | xargs -0 grep -l "$1" 2>/dev/null \
        | xargs -r ls -1t 2>/dev/null | head -1
}

online_bots() { # $1=chardb $2=prefix
    $DB "SELECT COUNT(*) FROM $1.characters c JOIN acore_auth.account a ON c.account=a.id WHERE a.username REGEXP '^$2[0-9]+$' AND c.online=1;" 2>/dev/null
}

events=()
report=""
for s in "${spec[@]}"; do
    IFS='|' read -r ID MODEL CHARDB PREFIX <<< "$s"
    TAG="i$ID"
    P=$(trainer_pid "$MODEL")
    WS=$(ws_count "$ID")
    BOTS=$(online_bots "$CHARDB" "$PREFIX")
    L=$(latest_log "$MODEL")

    ent_coef=""; entropy_loss=""; fps=""; evar=""; tts=""
    if [ -n "$L" ]; then
        ent_coef=$(grep -a '\[entropy\]' "$L" | tail -1 | grep -aoE 'ent_coef=[0-9.]+' | cut -d= -f2)
        entropy_loss=$(grep -aoE 'entropy_loss\s+\|\s+-?[0-9.e]+' "$L" | tail -1 | awk '{print $3}')
        fps=$(grep -aoE 'rollout_fps=[0-9]+' "$L" | tail -1 | cut -d= -f2)
        evar=$(grep -aoE 'explained_variance\s+\|\s+-?[0-9.e]+' "$L" | tail -1 | awk '{print $3}')
        tts=$(grep -aoE 'total_timesteps\s+\|\s+[0-9]+' "$L" | tail -1 | awk '{print $3}')
    fi

    # policy entropy (nats) = -entropy_loss (SBX logs raw -mean(entropy), max=log41=3.71)
    H=""
    if [ -n "${entropy_loss:-}" ]; then
        H=$(python3 -c "print(f'{-float('${entropy_loss}'):.3f}')" 2>/dev/null)
    fi

    EPS=$( $DB "SELECT CONCAT_WS('|', COUNT(*), ROUND(AVG(reward),4), ROUND(AVG(kill_count),4), ROUND(AVG(length),0), ROUND(AVG(xp),1)) FROM $CHARDB.neuralbot_episodes WHERE created_at > NOW() - INTERVAL 10 MINUTE;" 2>/dev/null )
    IFS='|' read -r n_ep avg_rw avg_kill avg_len avg_xp <<< "$EPS"

    report+="$TAG[t=${tts:-?} fps=${fps:-?} H=${H:-?} entc=${ent_coef:-?} ev=${evar:-?} ep/10m=${n_ep:-0} rew=${avg_rw:-?} kill=${avg_kill:-?} len=${avg_len:-?} xp=${avg_xp:-?} bots=${BOTS:-0} ws=$WS] "

    [ -z "$P" ] && events+=("$TAG trainer DOWN")
    [ "${BOTS:-0}" -lt 790 ] && events+=("$TAG bots=${BOTS:-0} <790")
    [ "$WS" = "0" ] && events+=("$TAG worldserver DOWN")
    if [ -n "$ent_coef" ]; then
        python3 -c "import sys; sys.exit(0 if float('$ent_coef') <= 0.00055 else 1)" 2>/dev/null \
            && events+=("$TAG reached entropy floor (ent_coef=$ent_coef)")
    fi
done

echo "[$TS] $report" >> "$LOG"
echo "[$TS] $report"
if [ ${#events[@]} -gt 0 ]; then
    printf 'EVENTS: %s\n' "${events[*]}" | tee -a "$LOG"
else
    echo "EVENTS: none"
fi
