# Changelog

All notable changes to `mod-neuralbot` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/) — currently pre-release (`0.x`).

## [0.6.0] — 2026-08-23

### Added
- **Level-1 baseline spells at creation** (`NeuralBotFactory::LearnBaselineSpells`).
  `Player::Create` auto-learns class/race spells via skill-line ability with
  `temporary=true` (player not in world), so `_SaveSpells` skipped them; at login
  `LearnDefaultSkills` skips re-learning because the *skills* were already saved — bots
  ended up with zero spells. The factory now converts `PLAYERSPELL_TEMPORARY → NEW`
  before the creation `SaveToDB` (400 bots now carry ~45 spells each) and explicitly
  learns the few rank-1 abilities outside the auto-learn set (e.g. warlock Summon Imp).
  Trainer-bought spells remain the ONLY path to higher ranks — no auto-maintenance.

### Changed
- **`spellLearned` is now a native reward term.** It was diagnostic-only (computed but
  not summed). Learning a spell is native character progression (self-limiting:
  `CanTeachSpell` blocks re-buying a known rank and money gates it), so it joins the
  native total — `xp + gold + level + questCompleted + spellLearned − death` — to
  shorten the otherwise-very-long walk→buy→cast→kill credit chain.
- **Trainer visibility measured:** 151/400 bots (38%) have a trainer within the 60 yd
  entity window (median 26.6 yd). The other 62% spawn too far out to discover trainers;
  a curriculum staging offset toward trainers is a future lever (ROADMAP §6).

### Fixed
- **Shutdown crash class eliminated.** Two related bugs:
  1. SIGABRT at exit — the static `NeuralBotWSHandler` destructor destroyed a joinable
     accept thread (`std::terminate`). Added an `OnShutdown` hook + destructor detach.
  2. The follow-up join-hang — `Stop()` used `pthread_timedjoin_np`, which reaped the
     thread behind `std::thread`'s back, so the destructor's later `detach()` hit a
     dangling pthread_t (use-after-free; intermittent across reboots). The legacy TCP
     debug handler is dead code (shm superseded it), so `NeuralBot.WebSocketPort` now
     defaults to **0 (disabled)** — no accept thread at all. When enabled, `Stop()`
     wakes the blocking `accept()` with a dummy connect and does a clean `join()`.
- **systemd watchdog was killing the stack every minute.** `neuralbot-watchdog.service`
  used the default `KillMode=control-group`, so every time `watchdog.sh` exited (after
  one pass) systemd SIGTERM'd everything in its cgroup — including the worldserver it
  just spawned (~72 s after each boot, a clean `Halting process…`). Now
  `KillMode=process` (reaps only the script); unit tracked in-repo at
  `scripts/neuralbot-watchdog.service`.
- **KL guard was pinning LR at its 1e-5 floor.** SBX 0.25.0 *does* serialize
  `target_kl` + `adaptive_lr` (contrary to the earlier diagnosis), so a warm-started
  model kept its stale guard braking at the floor. `NEURALBOT_KL=0` now fully disables
  the guard (explicitly clears both); `_setup_lr_schedule()` rebuilds the LR schedule
  after a retune; `adaptive_lr.max_learning_rate` is capped at the base LR so the guard
  can only brake, never overshoot (its default max is 1e-2 = 40× base).
  Watchdog now passes `NEURALBOT_ENT`/`NEURALBOT_KL` and defaults `NEURALBOT_LR=1.5e-4`.

## [0.5.0] — 2026-08-19

### Changed
- **Pipelined shm protocol (throughput architecture):** Python no longer sits in the
  step critical path. A harvester thread continuously copies frame batches into a
  depth-3 queue (clearing `obs_ready` immediately); the trainer consumes 1-tick-stale
  observations (standard frame-skip semantics) while C++ never idles waiting for
  Python. C++ gains an `ObservationsPending()` backpressure guard so frames are never
  torn mid-copy. Step rate is now min(C++, Python) with full overlap instead of the sum
  of latencies (C++ 18.5k/s, python 15.6k/s measured). Also retires the broken
  cross-process eventfd wait (was falling back to polling anyway).
- `MapUpdate.Threads` 3 → 5 in worldserver.conf.

### Fixed
- v8 lr 7e-4 instability follow-up: v9 runs lr 4e-4 + `target_kl=0.02` (KL-guard
  early-stop), warm-started from the v8 60M pre-collapse peak.

## [0.4.4] — 2026-08-18

### Fixed
- **Frame was 84% bot-clutter — hostiles invisible (v7 run).** All 400 bots share racial
  spawn points, so the distance-sorted 64-slot entity list was filled by ~54 friendly
  player-bots at 0-3 yd plus critters; the attackable mobs never entered the observation
  and TARGET_ENTITY_i pointed at clutter (0/400 bots could see a mob). Frame now excludes
  critters and caps players at the 4 nearest: 211/400 bots see an attackable mob
  (median 26 yd). Note: starter-area mobs read as reaction=NEUTRAL until attacked —
  attackable = creature && reaction != FRIENDLY.

## [0.4.3] — 2026-08-18

### Added
- **Curriculum staging (ROADMAP §7, config `NeuralBot.Curriculum`, default on):** at
  episode start, bots with no hostile within 60 yd are teleported (`NearTeleportTo`) to
  ~12 yd of the nearest hostile (120 yd scan, critters excluded). Collapses the ~190-step
  approach phase that kept the native XP reward outside any practical credit horizon;
  rewards stay 100% native — only the episode starting distribution changes. Disable to
  restore full-spawn difficulty once the combat loop is learned.

## [0.4.2] — 2026-08-18

### Changed
- **PPO credit horizon for sparse delayed reward (v6 run):** gamma 0.99 → 0.999,
  gae_lambda 0.95 → 0.98, ent_coef 0.05 → 0.02. Evidence from ~180M v5 steps: kills/ep
  rose 0.022 → 0.08 only to re-establish identically after a character reset (policy
  preserved) — i.e. the curve tracked bots spreading from the spawn clump, not learning;
  entropy never left maximum. Root cause: with gamma=0.99 the ~100-step credit horizon
  cannot span the ~190-step approach to a mob, so approach actions receive no credit for
  the kill they enable. 0.999 (~1000 steps) puts XP inside the window.
- Watchdog resumes model lineage v6.

## [0.4.1] — 2026-08-18

### Fixed
- **Idle timeout starved combat** (100M-step flat-reward root cause): at ~32 shm
  steps/s per bot, the 200-step idle budget ≈ 6 s wall time, while mobs sit 40+ yd
  from the spawn clumps (~190 walk steps). Episodes terminated mid-approach — only
  0.2% ever reached combat. Idle threshold 200 → 1500 (~47 s); `_maxSteps` stays 3000.
- `ATTACK_START` now auto-selects the nearest hostile when nothing is targeted
  (interim auto-service, same as the CAST fallback; ROADMAP §3 removes them once the
  policy works).
- Watchdog: training budget 100M → 1B (absolute in SB3 — resuming a finished model
  with the same budget made `learn()` exit instantly, restart-looping all night).

## [0.4.0] — 2026-08-18

### Changed
- **Action space v2 (ROADMAP §4 / DESIGN.md):** 16 → 41 discrete actions.
  - `MOVE_TO_TARGET` — MotionMaster `MoveChase` toward the selected unit (navmesh
    pathfinding); fallback chain: nearest hostile → nearest chest GO. Melee can finally
    close distance; the legacy 3-yard tank controls remain as fallbacks.
  - `TARGET_ENTITY_0..17` — select the i-th nearest entity *exactly as listed in the
    frame's distance-sorted entities[]* (guid cache filled by BuildFrame guarantees
    obs↔action index consistency).
  - `CAST_SPELL_0..7` — i-th spellbook entry in frame spells[] order (replaces the three
    auto-populated slots; enumeration mirrors BuildFrame exactly).
  - `TARGET_NEAREST_FRIENDLY` / `TARGET_NEAREST_CORPSE`, `ATTACK_STOP`.
  - `INTERACT_TARGET` — one context action on the selected unit: questgiver hello+accept,
    trainer buy (fixes ROADMAP §6 path), vendor browse, chest use. Gated at 5.5 yd like
    the real client — approach→interact must now be learned.
  - `COMPLETE_QUEST` / `LOOT` kept as context scans (auto-service removal tracked in §3).
- Per-action `LOG_INFO` → `LOG_DEBUG` (was ~13k lines/s → ~1 GB/hour of Server.log).
- Episode logging: act columns now generated from ACTION_COUNT; `neuralbot_episodes`
  extended to act_0..act_40 (live ALTER).

## [0.3.2] — 2026-08-18

### Fixed
- **Dead-bot episode loop.** Nothing in the environment revived a bot after death, so
  `ShouldTerminate` (!IsAlive) fired on *every* step: each dead bot emitted an endless
  stream of length-1 zero-reward episodes, permanently poisoning its env slot (~8% of the
  batch within the first minutes of the v4 run, growing). `NeuralBotMgr` now calls
  `NeuralBotInstance::ReviveIfDead()` on episode end — `ResurrectPlayer(0.5f)` +
  `SpawnCorpseBones()` at the death spot (same sequence mod-playerbots uses at spirit
  healers) — so the death penalty stays native and sparse, and the next episode starts
  immediately.

## [0.3.1] — 2026-08-18

### Fixed
- Bot name generation could emit names ending in `GM` (e.g. slot 194 → `NeuralbotGM`).
  AzerothCore reserves every name ending in `GM`/`gm` (`ObjectMgr::IsReservedName`/
  `IsProfanityName`), so `Player::LoadFromDB` set `AT_LOGIN_RENAME` and that bot could
  never spawn — permanently killing one env slot out of 400 on every boot. The letter-pair
  encoding now skips the `GM` pair entirely (slot 194 → `NeuralbotGN`).

### Added
- `scripts/archive_to_nas.sh` + systemd user timer (`neuralbot-archive.timer`): daily
  archival of large artifacts to the NAS (`~/NAS/temp/neuralbot`) — `neuralbot_episodes`
  rows older than 2 days (gzip SQL + DELETE), model checkpoints beyond the newest 5,
  old training logs, and `Server.log` once it passes 500 MB (grows ~1 GB/hour while
  training).

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
