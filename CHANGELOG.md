# Changelog

All notable changes to `mod-neuralbot` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/) — currently pre-release (`0.x`).

## [Unreleased]

### Planned
- Native reward (XP/level/gold/death/quest) — remove hand-crafted shaping terms.
- Faithful structured entity state (variable-length, transformer policy) replacing the fixed 85-float vector.
- Model-based RL (DreamerV3) as a successor/alternative to PPO.
- Reliable spell learning (friendly-targeting / trainer navigation).
- Curriculum across classes and starting zones.

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
