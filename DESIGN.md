# Design

Design of the `mod-neuralbot` rebuild: faithful state, native reward, faithful actions.
This supersedes the v1 "85 hand-picked floats + 16 tank-control actions + 14 shaped reward
terms" design. Reference: `README.md`, `ROADMAP.md`.

## Principles

1. **Faithful state.** The agent observes real, structured game state (entities, spells,
   quests, resources) with real values — not hand-normalized scalars.
2. **Native reward.** Reward is the game's own signal: XP, money, level, death, quest
   completion. No shaping terms (`kill`, `enemy_proximity`, `target_acquired`, etc.).
3. **No auto-services.** No auto-quest, no auto-target, no auto-loot. The agent must act.
4. **The environment is the curriculum.** One class/zone at a time; expand as mastered.
5. **Server-side moat.** We expose the complete true state from `Player`/`Unit`/`Creature`/
   `GameObject`/`Quest`/`SpellInfo` — what no vision or simplified-sim approach can.

## Ground truth (3.3.5)

| System | Value |
|--------|-------|
| Walk / Run / Run-back / Swim / Fly | 2.5 / 7.0 / 4.5 / 4.7222 / 7.0 yd·s⁻¹ |
| Turn rate | 3.1416 rad·s⁻¹ |
| Global cooldown | 1.5 s (1.0 s rogue/cat druid), floor 0.75 s w/ haste |
| Energy | max 100, regen 10·s⁻¹ |
| Combo points | max 5 |
| Resources | mana, rage, energy, runic power, focus |
| XP to level (1–10) | `40·x² + 360·x` → 400, 900, 1400, 2100, 2800, 3600, 4500, 5400, 6500, 7600 |
| XP to level (general) | `((8·CL) + Diff(CL)) · MXP(CL) · RF(CL)`; RF = 1 (≤10), `1−(CL−10)/100` (11–27), 0.82 (28–59), 1 (60+) |
| XP gain (mob kill) | `Acore::XP::Gain` (base 45 for 1–60 content; +xp mods for elite/group) |

AzerothCore already encodes these: `player_xp_for_level` (world DB), `Acore::XP::Gain`,
`Unit::GetSpeed`, `SpellInfo` cooldown/cost/range, `NPCFlags` / `GameobjectTypes` enums.

---

## State schema (v2)

Structured, entity-centric. Serialized over shared memory as a **fixed-size per-bot frame**
of packed fixed-width records (keeps zero-copy `numpy` feasible) with a header carrying the
*actual* counts. Real values, no normalization.

```
┌─ frame header ──────────────────────────────────────────────┐
│ self      (fixed)   health/max, resources, xp, money, pos,   │
│                     map/zone/area, flags, target guid, …     │
│ target    (fixed)   guid, entry, health/max, level, pos,     │
│                     reaction, alive, casting, distance       │
│ counts    (fixed)   n_spells, n_quests, n_entities, n_items  │
├─ spells   (≤64)     spellId, cooldown_ms, ready, cost, range │
├─ quests   (≤16)     questId, status, obj[0..3]               │
├─ entities (≤64)     guid, entry, type, level, health/max,    │
│                     dx, dy, distance, reaction, npcflags,    │
│                     alive, in_combat, casting                │
└─ items    (≤16)     guid, entry, quality, distance           │
```

### Entity typing

- **Creatures** (`Creature`) — `NPCFlags` distinguishes role: `GOSSIP`, `QUESTGIVER`,
  `TRAINER[_CLASS|_PROFESSION]`, `VENDOR[_AMMO|_FOOD|_POISON|_REAGENT]`, `REPAIR`,
  `FLIGHTMASTER`, `INNKEEPER`, `BANKER`, `AUCTIONEER`, `STABLEMASTER`, `MAILBOX`, …
- **Players** (`Player`) — includes the other NeuralBots and playerbots (true diversity).
- **GameObjects** (`GameObject`) — `GetGoType()`: `DOOR`, `CHEST`, `QUESTGIVER`, `GOOBER`,
  gathering nodes (mining/herbalism), `FISHINGHOLE`, …
- **Ground items** (`Item`/`WorldObject`) — lootable corpses/objects surface via `LOOT`.

### Reaction

`IsHostileTo`, `IsFriendlyTo`, else neutral — the real faction model, not a hardcoded flag.

### Why fixed-size packed records

- Zero-copy `numpy` views still possible per section (contiguous fixed-stride records).
- mmap region size stays precomputable (cap × record size × max bots) — the current shm
  design depends on this.
- No serialization dependency (protobuf/flatbuffers); Python parses with `np.frombuffer`.

---

## Action space (v2)

Replace "tank controls" (turn/forward/backward) with **point-navigation + targeted actions**.

### Interim (discrete, PPO-compatible)

| Family | Actions |
|--------|---------|
| Movement | `MOVE_TO_TARGET`, `STOP`, `MOVE_FORWARD`, `MOVE_BACKWARD` (legacy fallback) |
| Targeting | `TARGET_ENTITY[i]` (by frame index), `TARGET_NEAREST_ENEMY`, `TARGET_NEAREST_FRIENDLY`, `TARGET_NEAREST_CORPSE` |
| Combat | `ATTACK`, `CAST_SPELL[i]` (spellbook index), `CANCEL_CAST` |
| Interaction | `INTERACT` (gossip/quest/loot/vendor/trainer — context by target), `ACCEPT_QUEST`, `TURN_IN_QUEST`, `LOOT` |

`MOVE_TO_TARGET` uses `MotionMaster::MovePoint` (navmesh pathfinding) toward the selected
target, so the agent navigates the real world rather than wiggling with turns.

### Target (DreamerV3, mixed discrete + continuous)

- Continuous movement head: heading (rad) + throttle [-1,1] (forward/back) + strafe.
- Discrete heads: target-by-index, cast-by-index, interact.
- DreamerV3's action model handles mixed spaces natively (per-head distributions).

---

## Reward (native)

Scalar = weighted sum of the game's own signals. No shaping.

| Term | Source | Signal |
|------|--------|--------|
| `xp` | `PLAYER_XP` delta | leveling is the core objective |
| `money` | `GetMoney()` delta | gold earned from loot/quests/vendoring |
| `level` | `GetLevel()` delta | sparse bonus on level-up |
| `death` | `OnPlayerJustDied` | sparse penalty |
| `quest_complete` | `QUEST_STATUS_COMPLETE` transition | sparse bonus on turn-in |

Reward components are logged to `neuralbot_episodes` for analysis **only** — not summed.

Rationale: XP *is* the game's reward for leveling; money and quest completion are its other
two currencies. Everything else (kills, exploration, looting) is instrumental and must be
learned through these signals, not hand-shaped.

---

## Shared memory protocol (v2)

- Keep the existing directional flag protocol (`actions_ready` / `obs_ready` / `eventfd`).
- Replace `obs_flat[bot × 100 float]` with `frame[bot × FRAME_BYTES]` (header + packed
  records). Python still `mmap`s the region and slices per-section views.
- `SHM_VERSION` bumps to `2`; the control block carries the schema version so Python and
  C++ negotiate.
- Actions move from `uint8[4096]` to a per-bot action struct (target index + discrete
  action id + optional continuous fields), sized for v2.

---

## Migration phases

1. **Native reward** — delete shaping terms from `ComputeReward` (small, isolated).
2. **Structured state** — new frame schema in C++ (`BuildObservationInto` → builder) +
   Python parser (`shared_memory_env.py`).
3. **Auto-services off** — remove `AutoQuest`, auto-target, auto-loot paths.
4. **Action rework** — point-nav + indexed targeting + spellbook-index casting.
5. **DreamerV3** — swap policy; world model learns real dynamics.

Each phase is a separately buildable, testable commit (see `CHANGELOG.md`).
