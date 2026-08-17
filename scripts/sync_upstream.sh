#!/usr/bin/env bash
# Sync all three NeuralBot-related repos with their upstreams and push to the
# avirar forks. Non-destructive: fetch + merge (no rebase, no force-push).
# Run when the worldserver is NOT mid-training (it switches branches).
#
#   ./scripts/sync_upstream.sh
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # mod-neuralbot
ACORE_DIR="$(cd "$MODULE_DIR/../.." && pwd)"                     # azerothcore-wotlk
PLAYERBOTS_DIR="$ACORE_DIR/modules/mod-playerbots"

log() { echo "==> $*"; }
dirty() { [ -z "$(git status --porcelain)" ] || { echo "ERROR: $1 has uncommitted changes; commit or stash first" >&2; exit 1; }; }

# ── Parent: azerothcore (mod-playerbots fork) ──────────────────────────────
log "Syncing azerothcore (upstream -> Playerbot + neuralbot)"
cd "$ACORE_DIR"
git fetch upstream
git checkout Playerbot
git merge --ff-only upstream/Playerbot
git checkout neuralbot
git merge upstream/Playerbot -m "chore: sync upstream Playerbot branch"
git push origin Playerbot neuralbot

# ── Playerbots module ──────────────────────────────────────────────────────
log "Syncing mod-playerbots (upstream -> master)"
cd "$PLAYERBOTS_DIR"
git fetch upstream
git checkout master
git merge --ff-only upstream/master
git push origin master

# ── NeuralBot module (upstream is us; just push) ───────────────────────────
log "Pushing mod-neuralbot"
cd "$MODULE_DIR"
git push origin master

log "Sync complete."
