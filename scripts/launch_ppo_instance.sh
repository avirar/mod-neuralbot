#!/usr/bin/env bash
# Launch the PPO trainer for a given instance.
# Usage: scripts/launch_ppo_instance.sh <instance> [model_prefix]
#   instance 1 -> shm /neuralbot_shm,  model wow_neuralbot_model_v15_dense, decay 0.15 (MODERATE probe)
#   instance 2 -> shm /neuralbot_shm2, model wow_neuralbot_model_i2,       decay 0.25 (slow arm, fresh)
#   instance 3 -> shm /neuralbot_shm3, model wow_neuralbot_model_i3,       decay 0.05 (FAST arm, warm from i2)
set -euo pipefail
INST="${1:?usage: launch_ppo_instance.sh <1|2|3>}"
MODULE_DIR=/home/luke/GIT/azerothcore-wotlk/modules/mod-neuralbot
cd "$MODULE_DIR"
source .venv/bin/activate

case "$INST" in
    2) SHM=/dev/shm/neuralbot_shm2; MODEL="${2:-wow_neuralbot_model_i2}"; DECAY=0.5; DB=acore_characters2 ;;
    3) SHM=/dev/shm/neuralbot_shm3; MODEL="${2:-wow_neuralbot_model_i3}"; DECAY=0.5; DB=acore_characters3 ;;
    *) SHM=/dev/shm/neuralbot_shm; MODEL="${2:-wow_neuralbot_model_v15_dense}"; DECAY=0.5; DB=acore_characters ;;
esac

# resume from newest checkpoint if present
NEWEST=$(ls -1t "${MODEL}"*_steps.zip 2>/dev/null | head -1 || true)
if [ -n "$NEWEST" ]; then cp "$NEWEST" "${MODEL}.zip"; fi

NEURALBOT_MODEL="$MODEL" NEURALBOT_TIMESTEPS=1000000000 \
  NEURALBOT_LR=1.5e-4 NEURALBOT_ENT=0.01 NEURALBOT_ENT_FINAL=0.003 \
  NEURALBOT_ENT_DECAY_FRAC="$DECAY" NEURALBOT_KL=0 NEURALBOT_REWARD_MODE=symlog \
  NEURALBOT_BATCH_SIZE=4096 NEURALBOT_EPOCHS=5 NEURALBOT_NSTEPS=1024 \
  NEURALBOT_NUM_BOTS=800 SHM_PATH="$SHM" NEURALBOT_DB_NAME="$DB" \
  nohup python3 -u python/train_v3.py > "python/logs/train_i${INST}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "instance $INST PPO pid $!"
