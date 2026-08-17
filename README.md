# mod-neuralbot

Reinforcement-learning bots for [AzerothCore](https://www.azerothcore.org/) (World of Warcraft 3.3.5a / Wrath of the Lich King).

NeuralBot injects client packets **server-side** to control hundreds of bot characters in a live game world, exposes their observations and a scalar reward over a shared-memory bus, and trains a policy with [SBX PPO](https://github.com/araffin/sbx) (JAX) in Python.

> **Status:** experimental. The agent currently learns to level, fight, loot, and quest. Spell learning via trainers is wired up but not yet reliable (see [Known issues](#known-issues)).

---

## Design goals

- **Scale.** Hundreds of bots running concurrently in one worldserver, stepped as a single batch.
- **No client.** Everything happens on the server through packet injection; no game client, no rendering, no vision pipeline.
- **Real game.** The bots play the actual WotLK game — quest chains, spells, loot, zones — not a simplified simulation.
- **Learn, don't script.** Bots choose from a discrete action set; behavior emerges from reward, not hardcoded rules.

## Architecture

```
┌────────────────────────────  worldserver (C++)  ───────────────────────────┐
│                                                                            │
│  NeuralBotScripts  →  hooks (WorldScript / PlayerScript / PlayerbotScript) │
│        │                                                                   │
│  NeuralBotMgr  →  orchestrates login + per-tick batch stepping            │
│        │                                                                   │
│  NeuralBotInstance  →  per-bot: BuildObservationInto / ExecuteAction /     │
│                        ComputeReward / ResetRewardTracking                 │
│        │                                                                   │
│  NeuralBotSharedMem  →  /dev/shm/neuralbot_shm (mmap + eventfd)            │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │  shared memory (zero-copy)
┌──────────────────────────────────▼─────────────────────────────────────────┐
│  Python (SBX PPO, JAX)                                                     │
│                                                                            │
│  SharedMemoryVecEnv  →  VecEnv over the shm region                         │
│  train_v3.py         →  PPO policy, MySQL episode logging, checkpoints     │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data flow per step

1. Python writes 400 actions (`uint8`) into shared memory and sets `actions_ready = 1`.
2. The C++ world thread applies each action to its bot, then writes observations, reward, and done flags.
3. C++ sets `obs_ready = 1` and signals an `eventfd`.
4. Python wakes, reads the batch zero-copy as `numpy` arrays, returns `(obs, rewards, dones, infos)`.

### Components

| Path | Role |
|------|------|
| `src/NeuralBotFactory.cpp` | Creates bot accounts (`nbot*`) and characters (40 race/class starter combos) |
| `src/NeuralBotMgr.cpp` | Batch stepping, shared-memory orchestration, login queue |
| `src/NeuralBotInstance.cpp` | Per-bot observation/reward/action implementation |
| `src/NeuralBotSharedMem.cpp` | `/dev/shm/neuralbot_shm` mmap region |
| `src/NeuralBotScripts.cpp` | Registers world/player/playerbot script hooks |
| `src/NeuralBotWSHandler.cpp` | Legacy WebSocket/TCP control server (superseded by shm) |
| `python/train_v3.py` | Current training entry point (SBX PPO + shm + MySQL) |
| `python/shared_memory_env.py` | `SharedMemoryVecEnv` (VecEnv over shm) |
| `python/neuralbot_client.py` | Obs/action constants and legacy TCP client |

## Observation space (85 floats)

| Group | Size | Offsets | Contents |
|-------|------|---------|----------|
| `playerState` | 20 | 0–19 | health, mana, rage, level, position, orientation, money, zone, target (presence/health/level/dist/hostile), alive, moving, XP, XP progress, map |
| `nearbyUnits` | 5 × 8 | 20–59 | 5 nearest unfriendly units: health, level, distance, hostile, in-combat, is-player, entry, alive |
| `combatState` | 15 | 60–74 | in-combat, casting, stunned, dead, stand state, 5 spell-slot cooldown flags, trainer info (dist, in-range, learnable, affordable, present) |
| `questState` | 10 | 75–84 | quest count, nearest quest-giver/ender distance, completable, best progress, top-4 progress |

Observation constants are defined in `src/NeuralBotCommon.h` and mirrored in `python/neuralbot_client.py`.

## Action space (16 discrete actions)

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | `NOOP` | 8 | `CAST_SPELL_1` |
| 1 | `MOVE_FORWARD` | 9 | `CAST_SPELL_2` |
| 2 | `MOVE_BACKWARD` | 10 | `CAST_SPELL_3` |
| 3 | `TURN_LEFT` | 11 | `INTERACT_NPC` |
| 4 | `TURN_RIGHT` | 12 | `COMPLETE_QUEST` |
| 5 | `STOP_MOVE` | 13 | `TARGET_QUEST_GIVER` |
| 6 | `TARGET_NEAREST_ENEMY` | 14 | `LOOT` |
| 7 | `ATTACK_START` | 15 | `LEARN_SPELL` |

## Reward (native, v0.2.0)

Computed per step in `NeuralBotInstance::ComputeReward`. The scalar `total` is what PPO
optimizes, and is **native only** — the game's own signal, no shaping:

```
total = xpDelta + money(lootReward) + levelReward + questCompleted − deathPenalty
```

The remaining components are computed but **diagnostic-only** (logged to
`neuralbot_episodes` for analysis, not summed). Level-ups are detected via
`PLAYER_NEXT_LEVEL_XP` so a level boundary does not read as a negative XP delta.

| Component | Summed? | Meaning |
|-----------|---------|---------|
| `xpDelta` | ✅ native | XP gained since last step |
| `lootReward` | ✅ native | Money (gold) delta from loot/quests/vendoring |
| `deathPenalty` | ✅ native | Bot died (sparse) |
| `questCompleted` | ✅ native | Quest turned in (sparse) |
| `levelReward` | ✅ native | Level-up milestone (sparse) |
| `damageTaken` | ❌ diagnostic | Damage received |
| `killReward` | ❌ diagnostic | Enemy killed |
| `questAccepted` | ❌ diagnostic | Accepted a quest |
| `questProximity` | ❌ diagnostic | Distance to quest giver/ender |
| `questProgress` | ❌ diagnostic | Quest objective progress |
| `enemyProximity` | ❌ diagnostic | Distance to nearest enemy |
| `targetAcquired` | ❌ diagnostic | Acquired a new target |
| `spellLearned` | ❌ diagnostic | Learned a new spell |
| `trainerProximity` | ❌ diagnostic | Distance to trainer |
| `timePenalty` | ❌ (disabled) | Constant small negative (`0.0`) |

## Shared memory layout

`/dev/shm/neuralbot_shm` — region size derived from `SHM_MAX_BOTS = 4096`.

| Offset | Field |
|--------|-------|
| `0x0000` | control block (magic, version, bot count, step count, eventfd, `actions_ready`, `obs_ready`, `shutdown`) |
| `0x0080` | `actions[4096]` (`uint8`) |
| `0x2000` | `obs_flat[4096 × 100]` (`float32`) — per bot: 85 obs + total reward + 14 components |
| end | `dones[4096]` (`uint8`) |

`SHM_OBS_PER_BOT = OBS_TOTAL_SIZE + 1 + 14 = 100`.

## Requirements

- AzerothCore WotLK (mod-playerbots fork, `Playerbot` branch) — required for the `PlayerbotScript` hook
- C++17 toolchain (CMake)
- Python 3.12 virtualenv at `.venv` (`--system-site-packages`)
- `sbx` (Stable-Baselines3 JAX), `stable-baselines3`, `gymnasium`, `numpy`, `pymysql`
- MySQL (`acore_characters`), CUDA GPU (JAX)

> **Toolchain note:** the bundled `deps/jemalloc` (5.2.1) does not compile with GCC 16
> (`std::__throw_bad_alloc` was removed). Upstream azerothcore already carries the fix
> (`throw std::bad_alloc()`); if you see that error, apply the same one-line change to
> `deps/jemalloc/src/jemalloc_cpp.cpp` or pull latest upstream.

## Building

Build the worldserver with the module included (same as the rest of AzerothCore):

```bash
cd /home/luke/GIT/azerothcore-wotlk
bash acore.sh compiler build
```

Enable the module in `env/dist/etc/modules/mod-neuralbot.conf`.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `NeuralBot.Enable` | 1 | Enable the module |
| `NeuralBot.BotCount` | 400 | Number of bot characters (10 per account) |
| `NeuralBot.BotCharacterName` | `Neuralbot` | Name prefix (`NeuralbotA`, `NeuralbotB`, …) |
| `NeuralBot.WebSocketPort` | 9000 | Legacy TCP control port |
| `NeuralBot.TickRateMs` | 50 | World-tick batching rate |
| `NeuralBot.MaxEpisodeSteps` | 1000 | Episode length cap |
| `NeuralBot.CleanupOnStartup` | 0 | Delete + recreate all bot accounts/chars on boot |
| `NeuralBot.AutoQuest` | 1 | Auto-accept/complete quests (see note) |
| `NeuralBot.LogPackets` | 0 | Verbose packet logging |

## Training

```bash
cd /home/luke/GIT/azerothcore-wotlk/modules/mod-neuralbot
source .venv/bin/activate
NEURALBOT_TIMESTEPS=20000000 python3 python/train_v3.py
```

- Model resumption: if `wow_neuralbot_model_v3.zip` exists, it loads and saves to `..._iter2`.
- Checkpoints every 2500 rollout steps (~1M global steps at 400 envs).
- Episode stats stream to `acore_characters.neuralbot_episodes`.

See `run_train.sh` for a wrapper that waits for bots to log in and launches training in the background.

### Episode database

`acore_characters.neuralbot_episodes` — one row per finished episode:

`episode, reward, length, xp, kill_count, death, quest_proximity, quest_progress, enemy_proximity, target_acquired, act_0 … act_15`

## Coexistence with mod-playerbots

NeuralBot is designed to run **alongside** `mod-playerbots` so the world contains both learned and scripted bot populations. The two modules use disjoint accounts/characters:

- NeuralBots → accounts `nbot0…`, characters `NeuralbotA…`
- Playerbots → accounts from `AiPlayerbot.RandomBotAccountPrefix`, characters `RndbotA…`

NeuralBot registers a `PlayerbotScript` (`OnPlayerbotPacketSent`) and therefore requires the playerbots fork's `Playerbot` branch. It filters strictly by its own bot set; playerbot randombots and altbots are unaffected.

## Known issues

- **Spell learning unreliable.** Bots find trainers but often fail the 5-yard interaction check (`ACTION_LEARN_SPELL` used heavily, zero spells learned in some runs). No friendly-targeting/navigation action exists to close the gap.
- **`neuralbot_client.py` / `NeuralBotWSHandler.cpp` are legacy.** The TCP path is superseded by shared memory and kept only for debug/status.
- **Fixed 85-float state.** The state is still a hand-picked flat vector; the structured, entity-centric schema is specified in `DESIGN.md` and tracked in `ROADMAP.md` (§1).

## Repository layout & git workflow

Three nested repos, all under `/home/luke/GIT/azerothcore-wotlk`:

| Repo | Local path | `origin` (your fork) | `upstream` | Branch |
|------|-----------|----------------------|------------|--------|
| AzerothCore (playerbots fork) | `…/azerothcore-wotlk` | `avirar/azerothcore-wotlk` | `mod-playerbots/azerothcore-wotlk` | `neuralbot` (ours), `Playerbot` (mirrors upstream) |
| Playerbots module | `…/modules/mod-playerbots` | `avirar/mod-playerbots` | `mod-playerbots/mod-playerbots` | `master` |
| NeuralBot module | `…/modules/mod-neuralbot` | `avirar/mod-neuralbot` | — | `master` |

- `modules/` is gitignored in the parent repo, so the two modules are independent repos.
- Our parent-repo changes (`conf/dist/config.sh` build type, a local DB-migration tweak)
  live on the `neuralbot` branch; `Playerbot` is a clean mirror of upstream.
- Weekly sync (`scripts/sync_upstream.sh`): fetch + merge `upstream` → push both branches.

## License

Part of AzerothCore (GPL v2). See the root `LICENSE`.
