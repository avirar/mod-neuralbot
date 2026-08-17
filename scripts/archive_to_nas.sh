#!/usr/bin/env bash
# Archive large NeuralBot artifacts to the NAS so flash storage stays lean.
# Run daily from cron (see AGENTS.md). Safe to run while training is active.
#
#  1. neuralbot_episodes rows older than EPISODES_KEEP_DAYS  -> gzipped SQL on NAS, then DELETE
#  2. model checkpoints beyond the newest KEEP_CHECKPOINTS   -> moved to NAS
#  3. training logs older than LOGS_KEEP_DAYS                -> moved to NAS
#  4. env/dist/bin/Server.log if > SERVER_LOG_MAX_MB         -> copied+gzipped to NAS, truncated
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NAS_DIR="${NAS_DIR:-$HOME/NAS/temp/neuralbot}"
STAMP="$(date +%Y%m%d_%H%M%S)"

EPISODES_KEEP_DAYS="${EPISODES_KEEP_DAYS:-2}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-5}"
LOGS_KEEP_DAYS="${LOGS_KEEP_DAYS:-7}"
SERVER_LOG_MAX_MB="${SERVER_LOG_MAX_MB:-500}"

DB_USER="${NEURALBOT_DB_USER:-acore}"
DB_PASS="${NEURALBOT_DB_PASS:-abc}"
DB_HOST="${NEURALBOT_DB_HOST:-127.0.0.1}"
DB_NAME="${NEURALBOT_DB_NAME:-acore_characters}"

mkdir -p "$NAS_DIR/episodes" "$NAS_DIR/checkpoints" "$NAS_DIR/logs"
log() { echo "[$(date '+%F %T')] $*"; }

# ── 1. Episodes table ────────────────────────────────────────────────────────
CUTOFF_DATE=$(date -d "-${EPISODES_KEEP_DAYS} days" '+%Y-%m-%d %H:%M:%S')
OLD_ROWS=$(mysql -u"$DB_USER" -p"$DB_PASS" -h "$DB_HOST" -N -e \
    "SELECT COUNT(*) FROM ${DB_NAME}.neuralbot_episodes WHERE created_at < '${CUTOFF_DATE}';" 2>/dev/null || echo 0)
if [ "${OLD_ROWS:-0}" -gt 0 ]; then
    OUT="$NAS_DIR/episodes/episodes_before_${STAMP}.sql.gz"
    mysqldump -u"$DB_USER" -p"$DB_PASS" -h "$DB_HOST" --no-create-info \
        --where="created_at < '${CUTOFF_DATE}'" "$DB_NAME" neuralbot_episodes 2>/dev/null | gzip > "$OUT"
    mysql -u"$DB_USER" -p"$DB_PASS" -h "$DB_HOST" -e \
        "DELETE FROM ${DB_NAME}.neuralbot_episodes WHERE created_at < '${CUTOFF_DATE}';" 2>/dev/null
    log "episodes: archived+deleted $OLD_ROWS rows (kept < ${CUTOFF_DATE}) -> $OUT"
else
    log "episodes: nothing older than ${CUTOFF_DATE}"
fi

# ── 2. Model checkpoints ─────────────────────────────────────────────────────
cd "$MODULE_DIR"
ls -1t wow_neuralbot_*_steps.zip 2>/dev/null | tail -n +$((KEEP_CHECKPOINTS + 1)) | while read -r f; do
    mv "$f" "$NAS_DIR/checkpoints/"
    log "checkpoint: moved $f"
done

# ── 3. Training logs ─────────────────────────────────────────────────────────
find "$MODULE_DIR/python/logs" -name '*.log' -mtime +"$LOGS_KEEP_DAYS" -print0 2>/dev/null |
    while IFS= read -r -d '' f; do mv "$f" "$NAS_DIR/logs/"; log "log: moved $f"; done

# ── 4. Worldserver log ───────────────────────────────────────────────────────
SERVER_LOG="/home/luke/GIT/azerothcore-wotlk/env/dist/bin/Server.log"
if [ -f "$SERVER_LOG" ]; then
    SIZE_MB=$(( $(stat -c%s "$SERVER_LOG") / 1024 / 1024 ))
    if [ "$SIZE_MB" -gt "$SERVER_LOG_MAX_MB" ]; then
        gzip -c "$SERVER_LOG" > "$NAS_DIR/logs/Server_${STAMP}.log.gz"
        truncate -s 0 "$SERVER_LOG"   # safe for the running append-only writer
        log "Server.log: archived ${SIZE_MB}MB and truncated"
    else
        log "Server.log: ${SIZE_MB}MB (under ${SERVER_LOG_MAX_MB}MB cap)"
    fi
fi

log "done"
