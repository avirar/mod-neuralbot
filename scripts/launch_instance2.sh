#!/usr/bin/env bash
# Launch the SECOND worldserver instance (800 bots, own shm + character DB).
# Instance 1 is the default `env/dist/etc/worldserver.conf` (shm /neuralbot_shm,
# accounts nbot*, chars DB acore_characters). This instance uses:
#   - config:  ../etc/instance2/worldserver.conf  (port 8086, SOAP 7879, RealmID 2,
#              CharacterDatabaseInfo -> acore_characters2)
#   - shm:     /neuralbot_shm2
#   - accounts: xbot*   chars: Xbot*  (in acore_characters2)
# The module config dir is compile-time fixed, so per-instance module values are set via
# AC_* env vars (IniKeyToEnvVarKey: NeuralBot.ShmName -> AC_NEURAL_BOT_SHM_NAME).
set -euo pipefail
REPO=/home/luke/GIT/azerothcore-wotlk
BIN="$REPO/env/dist/bin"

cd "$BIN"
# stdin MUST stay open: worldserver's console reader spin-loops on EOF
# ("AC>" flood, ~2 cores wasted). `tail -f /dev/null |` holds it open forever.
AC_NEURAL_BOT_SHM_NAME="/neuralbot_shm2" \
AC_NEURAL_BOT_BOT_CHARACTER_NAME="Xbot" \
AC_NEURAL_BOT_BOT_ACCOUNT_PREFIX="xbot" \
setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver -c ../etc/instance2/worldserver.conf' > ws2.out 2>&1 &
echo "instance2 worldserver pid $!"
