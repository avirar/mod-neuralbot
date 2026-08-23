# mod-neuralbot

Reinforcement-learning bots for [AzerothCore](https://www.azerothcore.org/) (World of Warcraft 3.3.5a / Wrath of the Lich King).

NeuralBot injects client packets **server-side** to control hundreds of bot characters in a live game world, exposes their observations and a scalar reward over a shared-memory bus, and trains a policy with [SBX PPO](https://github.com/araffin/sbx) (JAX) in Python.

> **Status:** experimental. The agent currently learns to level, fight, loot, and quest. Bots are born with their level-1 baseline spells (v0.6.0); higher ranks are trainer-bought and must be learned by the policy (see [Known issues](#known-issues)).

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

## Observation space (structured frame, v0.3.0)

The observation is a faithful, entity-centric frame (`src/NeuralBotFrame.h`), exchanged
over shared memory as packed records. Real game values, no normalization.

| Section | Count | Contents |
|---------|-------|----------|
| `self` | 1 | guid, level, health/mana/resource + maxes, xp/next-level-xp, money, pos+orientation, map/zone/area, alive/inCombat/moving/casting/inWater/mounted, class/race, combo points, target guid |
| `target` | 1 | guid, entry, type, health/max, level, dx/dy/dz, distance, reaction, flags, npcFlags |
| `spells` | ≤64 | spellId, cooldownMs, ready, cost, range/minRange, castTimeMs (full spellbook) |
| `quests` | ≤16 | questId, status, objective counters |
| `entities` | ≤64 | nearby creatures/players/gameobjects, typed with real faction reaction |
| `items` | ≤16 | nearby lootable corpses + chest gameobjects |

The wire layout is packed (`FRAME_BYTES = 5909`), `SHM_VERSION = 2`, with the control
block carrying `frame_bytes` for schema negotiation. Byte-accurate sizes are enforced by
`static_assert` in C++ and a `dtype.itemsize` check in Python.

For the SBX PPO `MlpPolicy`, `SharedMemoryVecEnv` projects each frame to a fixed tensor
(`OBS_FLAT_SIZE = 1148`). A transformer/DreamerV3 policy (ROADMAP §5) will consume the
structured records directly. The legacy 85-float vector still exists for the TCP
`NeuralBotWSHandler` path (unused by training).

## Action space (41 discrete actions, v0.4.0)

Point-navigation + entity-index targeting + spellbook-index casting. Entity/action
indices are consistent with the distance-sorted `entities[]` and `spells[]` frame sections.

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | `NOOP` | 20–27 | `CAST_SPELL_0..7` (i-th spellbook entry) |
| 1 | `MOVE_TO_TARGET` (navmesh; fallback nearest hostile) | 28 | `ATTACK_START` |
| 2 | `STOP_MOVE` | 29 | `ATTACK_STOP` |
| 3–6 | `MOVE_FORWARD/BACKWARD/TURN_LEFT/RIGHT` (legacy) | 30–37 | `TARGET_ENTITY_0..17` (i-th nearest entity) |
| 7 | `TARGET_NEAREST_ENEMY` | 38 | `INTERACT_TARGET` (quest/trainer/vendor/chest, gated 5.5 yd) |
| 8 | `TARGET_NEAREST_FRIENDLY` | 39 | `COMPLETE_QUEST` |
| 9 | `TARGET_NEAREST_CORPSE` | 40 | `LOOT` |

`TARGET_ENTITY_0` is ID 10, `TARGET_ENTITY_17` is ID 27; `CAST_SPELL_0` is ID 20,
`CAST_SPELL_7` is ID 27. See `NeuralBotCommon.h` `enum NeuralBotAction` for the canonical
list.

## Reward (native, v0.2.0 + v0.6.0)

Computed per step in `NeuralBotInstance::ComputeReward`. The scalar `total` is what PPO
optimizes, and is **native only** — the game's own signal, no shaping:

```
total = xpDelta + money(lootReward) + levelReward + questCompleted + spellLearned − deathPenalty
```

`spellLearned` (v0.6.0) is native character progression — self-limiting (a known rank
can't be re-bought; money gates it), so it can't be gamed and it shortens the
walk→buy→cast→kill credit chain. The remaining components are computed but
**diagnostic-only** (logged to `neuralbot_episodes` for analysis, not summed). Level-ups
are detected via `PLAYER_NEXT_LEVEL_XP` so a level boundary does not read as a negative
XP delta.

| Component | Summed? | Meaning |
|-----------|---------|---------|
| `xpDelta` | ✅ native | XP gained since last step |
| `lootReward` | ✅ native | Money (gold) delta from loot/quests/vendoring |
| `deathPenalty` | ✅ native | Bot died (sparse) |
| `questCompleted` | ✅ native | Quest turned in (sparse) |
| `levelReward` | ✅ native | Level-up milestone (sparse) |
| `spellLearned` | ✅ native | Learned a new spell (v0.6.0) |
| `damageTaken` | ❌ diagnostic | Damage received |
| `killReward` | ❌ diagnostic | Enemy killed |
| `questAccepted` | ❌ diagnostic | Accepted a quest |
| `questProximity` | ❌ diagnostic | Distance to quest giver/ender |
| `questProgress` | ❌ diagnostic | Quest objective progress |
| `enemyProximity` | ❌ diagnostic | Distance to nearest enemy |
| `targetAcquired` | ❌ diagnostic | Acquired a new target |
| `trainerProximity` | ❌ diagnostic | Distance to trainer |
| `timePenalty` | ❌ (disabled) | Constant small negative (`0.0`) |

## Shared memory layout

`/dev/shm/neuralbot_shm` — region size derived from `SHM_MAX_BOTS = 4096`.

| Offset | Field |
|--------|-------|
| `0x0000` | control block (magic, version, bot count, step count, eventfd, `actions_ready`, `obs_ready`, `shutdown`, `frame_bytes`) |
| `0x0080` | `actions[4096]` (`uint8`) |
| `0x2000` | `frames[4096]` (`NeuralBotFrame`, packed `FRAME_BYTES` = 5909 each) |
| end | `dones[4096]` (`uint8`) |

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
| `NeuralBot.WebSocketPort` | 0 | Legacy TCP control port (0 = disabled; superseded by shm) |
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

- **Trainer navigation not yet learned.** Bots are born with level-1 baseline spells
  (v0.6.0), but higher ranks require walking to a class trainer and buying them
  (`INTERACT_TARGET` → `CMSG_TRAINER_BUY_SPELL`). The wiring is complete (trainers appear
  in `entities[]` with the trainer NPC flag; ~38% of bots have one within 60 yd), but the
  policy has not yet learned the multi-step sequence. A curriculum staging offset toward
  trainers is a future lever (ROADMAP §6).
- **`neuralbot_client.py` / `NeuralBotWSHandler.cpp` are legacy.** The TCP path is
  superseded by shared memory and is **disabled by default** (`NeuralBot.WebSocketPort = 0`)
  after it caused shutdown crashes. Kept only for debug/status.
- **Ground items are approximated.** The frame `items` section currently lists corpses + chest gameobjects, not individual `Item` world objects (tracked in ROADMAP §1 follow-up).
- **MLP interim.** The policy still consumes a flattened projection of the structured frame; a transformer (DreamerV3) will consume the records directly (ROADMAP §5).

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
