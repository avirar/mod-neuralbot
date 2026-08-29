#!/usr/bin/env bash
# Launch the THIRD worldserver instance (800 bots, own shm + character DB).
#   - config:  ../etc/instance3/worldserver.conf  (port 8087, SOAP 7880, RealmID 3,
#              CharacterDatabaseInfo -> acore_characters3)
#   - shm:     /neuralbot_shm3
#   - accounts: ybot*   chars: Ybot*  (in acore_characters3)
# Per-instance module values via AC_* env vars (compile-time-fixed config dir).
# stdin held open via `tail -f /dev/null |` (prevents console EOF spin loop).
set -euo pipefail
REPO=/home/luke/GIT/azerothcore-wotlk
BIN="$REPO/env/dist/bin"

cd "$BIN"
AC_NEURAL_BOT_SHM_NAME="/neuralbot_shm3" \
AC_NEURAL_BOT_BOT_CHARACTER_NAME="Ybot" \
AC_NEURAL_BOT_BOT_ACCOUNT_PREFIX="ybot" \
setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver -c ../etc/instance3/worldserver.conf' > ws3.out 2>&1 &
echo "instance3 worldserver pid $!"
