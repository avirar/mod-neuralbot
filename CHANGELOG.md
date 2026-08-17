# Changelog

All notable changes to `mod-neuralbot` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/) — currently pre-release (`0.x`).

## [Unreleased]

### Planned
- Model-based RL (DreamerV3) as a successor/alternative to PPO, consuming the structured frame records directly.
- Reliable spell learning (friendly-targeting / trainer navigation).
- Curriculum across classes and starting zones.
- Remove auto-services (AutoQuest / auto-target / auto-loot).
- True ground-item (`Item`) scan in the frame `items` section (currently corpses + chest gameobjects).

## [0.3.0] — 2026-08-18

### Added
- **Faithful structured observation frame** (`src/NeuralBotFrame.h`), replacing the flat 85-float vector on the shared-memory path. Per-bot frame is packed (`#pragma pack(1)`, 5909 bytes):
  - `self` (96 B) — guid, level, health/mana/resource + maxes, xp/next-level-xp, money, pos+orientation, map/zone/area, alive/inCombat/moving/casting/inWater/mounted, class/race, combo points, target guid.
  - `target` (49 B) — guid, entry, type, health/max, level, dx/dy/dz, distance, reaction, flags, npcFlags.
  - `counts` (8 B) — n_spells/n_quests/n_entities/n_items.
  - `spells[64]` (28 B each) — spellId, cooldownMs, ready, cost, range/minRange, castTimeMs (full spellbook).
  - `quests[16]` (16 B each) — questId, status, obj[0..3] counters.
  - `entities[64]` (52 B each) — nearby creatures/players/gameobjects, typed (`NB_ENTITY_TYPE_*`) with real faction `reaction` (`NB_REACTION_*`); for gameobjects `npcFlags` carries `GetGoType()`.
  - `items[16]` (20 B each) — nearby lootable corpses + chest gameobjects.
  - `reward` tail (60 B) — native total + 14 diagnostic components.
- Real values, no normalization. Entity/action semantics are now available for the entity-index actions (ROADMAP §4).
- `SHM_VERSION = 2` and the control block carries `frame_bytes` for cross-language schema negotiation (C++ `static_assert` + Python `dtype.itemsize` check).
- `NeuralBotInstance::StepFrame`/`ResetFrame`/`BuildFrame` alongside the legacy `Step`/`Reset`.

### Changed
- Shared-memory obs region: `float[4096×100]` → packed `NeuralBotFrame[4096]` (`SHM_FRAME_BYTES` = 5909, ~24 MB region).
- `SharedMemoryVecEnv` parses structured frames via numpy structured dtypes and exposes a fixed flattened projection (`OBS_FLAT_SIZE` = 1148) for `MlpPolicy`. A transformer/DreamerV3 policy (§5) will consume the records directly.
- `NeuralBotMgr::ProcessSharedMemoryStep` writes frames via `StepFrame`; reward components are serialized once in `WriteFrameReward`.

### Kept
- Legacy 85-float `NeuralBotObservation`/`ToFloatArray` and the TCP `NeuralBotWSHandler` still compile (unused by training; removal tracked in ROADMAP §9).

## [0.2.0] — 2026-08-17

### Added
- Native reward. The scalar `total` is now `XP + gold + level milestone + quest completion − death`. Shaping terms (`killReward`, `damageTaken`, `questAccepted`, `questProgress`, `questProximity`, `enemyProximity`, `targetAcquired`, `trainerProximity`, `spellLearned`) are still computed but kept **diagnostic-only** (logged to `neuralbot_episodes`, not summed).
- XP level-up carry-over handling: reconstructs the true XP gain across a level boundary using `PLAYER_NEXT_LEVEL_XP`, so a level-up no longer reads as a large negative XP delta.
- `DESIGN.md` — faithful state schema, action-space rework, shared-memory v2, migration phases.

### Changed
- `ComputeReward` no longer sums shaping terms or the `-0.001` time penalty (`out.timePenalty = 0`).

## [0.1.0] — 2026-05-11

### Added
- `ACTION_LEARN_SPELL` (action 16), `OBS_COMBAT_STATE_SIZE` 10 → 15, `OBS_TOTAL_SIZE` 80 → 85.
- Trainer observations in `combatState[10..14]` (distance, in-range, learnable, affordable, present).
- `PLAYERHOOK_ON_LEARN_SPELL` hook and `+10` spell-learned reward plus trainer-proximity reward.
- `act_15` column in episode stats.

### Changed
- Shared-memory observation layout grew to `SHM_OBS_PER_BOT = OBS_TOTAL_SIZE + 1 + 14 = 100` floats per bot.

## [0.0.7] — 2026-05-11

### Added
- `ACTION_LOOT` (action 14) and money-delta loot reward.
- `act_14` column in episode stats.

### Changed
- Idle threshold relaxed: `0.01 → 0.001` over `50 → 200` steps.
- `MaxEpisodeSteps` raised `1000 → 3000`.
- `ent_coef` raised `0.02 → 0.05`.

### Fixed
- Graceful action-space mismatch when resuming a model with a different action count.

## [0.0.6] — 2026-05-11

### Added
- Model resume support for iterative training (loads previous `.zip`, saves to `_iter2`).

## [0.0.5] — 2026-05-11

### Fixed
- `ResetRewardTracking()` now runs in `ProcessSharedMemoryStep()` on episode end — previously bots were stuck in a permanent `done` state (all episodes length 1).
- `reward_components` are now always included in Python `infos` (removed `if not dones[i]` guard) — kill/xp/death data reach MySQL.

## [0.0.4] — 2026-05-11

### Added
- MySQL episode storage (`acore_characters.neuralbot_episodes`) replacing CSV.

### Fixed
- `BufferError` on env close (release numpy views before closing mmap).
- Checkpoint save frequency corrected.

## [0.0.3] — 2026-05-11

### Added
- Shared-memory IPC (`/dev/shm/neuralbot_shm`, mmap + eventfd) replacing TCP for batch stepping — single-threaded, zero-copy, ~15 ms/step.

### Changed
- `SharedMemoryVecEnv` replaces `ThreadedVecEnv`.

## [0.0.2] — 2026-05-11

### Added
- Parallel login and `ThreadedVecEnv`.
- SBX PPO (JAX) training at ~9K fps on GPU.

## [0.0.1] — 2026-05-10

### Added
- Quest-aware observation, reward, and action system.
- Enemy-proximity reward with 60-yard scan; cumulative CSV logging.
- Dynamic spellbook discovery and auto-population; auto-target nearest enemy when casting.
- Serialized `Step`/`Reset` through the world thread via a promise queue.
- Account/character creation fixes and `CleanupOnStartup`.
- Multi-bot architecture (20 bots across 2 accounts) → scalable dynamic bots.

### Fixed
- TCP error handling, stable training loop with env recreation, Python 3.12 compatibility.

## [0.0.0] — 2026-05-10

### Added
- Initial RL bot framework: TCP bridge, server-side packet tap, PPO training.
