# Roadmap

Long-horizon direction for `mod-neuralbot`. Tracks the architectural pivot away from
hand-crafted reward shaping toward a faithful-environment + native-reward design.

Priorities: **P0** = must do before next major iteration, **P1** = should do soon,
**P2** = later / exploratory.

---

## 0. Direction (why this matters)

The current design (85 hand-picked floats, 16 discrete actions, ~14 hand-tuned reward
terms, scripted auto-services) trains *fast* but is reward engineering, not a faithful
environment. The goal is a model that learns to play the real game from the real game's
own signal (XP, level, gold, death, quest completion) against a faithful, structured
state — with **no hardcoded hacks**.

AzerothCore gives us something no one else has: **server-side access to the complete,
true game state** and the ability to run **hundreds of parallel bots** in the real game.
That — not the GPU — is the moat. The 5090 accelerates the world model.

---

## 1. Faithful state representation — P0 — ✅ DONE (2026-08-18)

Replace the fixed 85-float vector with a structured, entity-centric observation the
agent can actually reason over.

- [x] Variable-length entity list: self, target, nearby units, nearby NPCs, quests,
      loot, trainers — each with type-tagged features. Emitted as capped packed records
      over shm (`self`/`target`/`counts` + `spells[64]` + `quests[16]` + `entities[64]`
      + `items[16]` + reward tail, `src/NeuralBotFrame.h`).
- [x] Emit *real* values (health, position, level, faction, NPC flag, quest state) —
      not normalized hand-picked scalars. Gameobjects typed by `GetGoType()`; reaction
      from the real faction model (`IsHostileTo`/`IsFriendlyTo`).
- [x] Structured message format over shared memory (packed binary layout) replacing the
      flat `float[4096×100]` region. `SHM_VERSION = 2`; control block carries `frame_bytes`.
- [x] Python parses into structured records (`np.frombuffer` + dtypes) and projects to a
      fixed tensor for `MlpPolicy` (transformer consumes records directly in §5).

## 2. Native reward — P0 — ✅ DONE (2026-08-17)

Remove shaping. Keep only the game's own signal.

- [x] XP gained → reward (level as a sparse bonus on top; level-up carry-over handled).
- [x] Gold/money delta.
- [x] Death penalty.
- [x] Quest completion (sparse bonus on turn-in).
- [x] Remove from total: `killReward`, `enemyProximity`, `targetAcquired`, `questProximity`,
      `trainerProximity`, `spellLearned`, hand-tuned `timePenalty` (kept as diagnostics).
- [x] Keep reward components in `infos` for analysis only (not summed into reward).

## 3. Kill the auto-services — P0

Let the agent act; stop scripting the world for it.

- [ ] Remove `AutoQuest` (auto-accept/auto-complete) — quests become agent actions.
- [ ] Remove auto-target nearest enemy on spell cast.
- [ ] Remove auto-loot scanning.
- [ ] Keep the services as *diagnostic* options only (default off).

## 4. Action space rework — P1

- [ ] Split movement from combat/targeting into a continuous or higher-fidelity action
      head (e.g. movement deltas, target by entity index, spell by ID).
- [ ] Spell selection by learned spellbook entry, not fixed `CAST_SPELL_1..3` slots.
- [ ] Add friendly-targeting action (required to fix spell learning, see §6).

## 5. Model-based RL (DreamerV3) — P1

- [ ] Evaluate official `danijar/dreamerv3` (JAX) against the shared-memory env.
- [ ] World model learns the *actual* game dynamics from server-side state — the
      differentiator no vision/sim approach can match.
- [ ] Keep PPO as a baseline; compare sample efficiency and final leveling speed.

## 6. Fix spell learning — P1

- [ ] Root cause: bots find trainers but fail the 5-yard interaction check.
- [ ] Add friendly-unit navigation/targeting so the bot can close distance.
- [ ] Verify `PLAYERHOOK_ON_LEARN_SPELL` fires and spells persist across episodes.

## 7. Curriculum — P2

- [ ] Start single class + single zone; add races/classes/zones as mastery improves.
- [ ] Gate by level bands (1–10, 10–20, …) so the policy sees consistent dynamics.
- [ ] Track per-zone/per-class win rates in `neuralbot_episodes`.

## 8. Coexistence with mod-playerbots — P1 (ongoing)

- [ ] Keep both populations running simultaneously ("true diversity" of AI players).
- [ ] Ensure disjoint accounts/characters and no packet-hook interference
      (NeuralBot filters strictly by its own bot set; `PlayerbotScript` shared safely).
- [ ] Long term: allow interaction (grouping, trading, PvP) between learned and
      scripted bots.

## 9. Infra / tooling — P2

- [ ] `epochs` loop in `train_v3.py` (currently single `model.learn()` call).
- [ ] Replace legacy `neuralbot_client.py` / `NeuralBotWSHandler.cpp` or retire them.
- [ ] Versioned observation/action schema negotiation in the shm control block.
- [ ] CI-style smoke test: build + 1-bot step round-trip.

---

## Status snapshot (2026-08-18)

- PPO + shared memory + MySQL: **done** (v0.1.0).
- Native reward (§2): **done** (v0.2.0). Shaping terms are now diagnostic-only.
- Faithful structured state (§1): **done** (v0.3.0). Packed entity-centric frame over
  shm v2; `MlpPolicy` consumes a flattened projection until DreamerV3.
- Research + design: **done** — `DESIGN.md` specifies the state schema, action rework,
  and shm v2.
- Remaining rebuild work, in order: §3 (kill auto-services), §4 (action rework),
  §5 (DreamerV3), §6 (spell learning).
- Earlier: iter 1–4 trained; kills/loot/xp all climbing; spell learning blocked.
