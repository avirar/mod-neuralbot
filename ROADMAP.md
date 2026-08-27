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

## 2. Native reward — P0 — ✅ DONE (2026-08-17, extended 2026-08-23)

Remove shaping. Keep only the game's own signal.

- [x] XP gained → reward (level as a sparse bonus on top; level-up carry-over handled).
- [x] Gold/money delta.
- [x] Death penalty.
- [x] Quest completion (sparse bonus on turn-in).
- [x] Spell learned (sparse bonus, v0.6.0) — native progression, self-limiting.
- [x] Remove from total: `killReward`, `enemyProximity`, `targetAcquired`, `questProximity`,
      `trainerProximity`, hand-tuned `timePenalty` (kept as diagnostics).
- [x] Keep reward components in `infos` for analysis (only the native terms are summed).

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

## 6. Fix spell learning — P1 — partial (2026-08-23)

- [x] Root cause: bots were created with **zero** spells — `Player::Create` learns
      class/race spells as temporary (not saved) and login skips re-learning. Fixed by
      converting `PLAYERSPELL_TEMPORARY → NEW` before the creation save (v0.6.0); bots
      now carry their level-1 baseline (~45 spells each).
- [x] `spellLearned` is a native reward term (v0.6.0) so trainer purchases are credited.
- [x] Trainer observation is in the frame: trainers appear in `entities[]` with the
      trainer NPC flag (151/400 bots have one within 60 yd, median 26.6 yd).
- [ ] Friendly-unit navigation to the trainer + `INTERACT_TARGET` → buy must still be
      *learned* by the policy (no auto-maintenance).
- [ ] Verify trainer-bought spells persist across episodes (the `OnPlayerLearnSpell` hook
      fires; spell state follows the normal save path).

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

## 10. Policy architecture (MoE + attention + hierarchy) — P0 (2026-08-26)

Three algorithms (PPO sparse, PPO dense `v15_dense`, R2-Dreamer) have now plateaued at
max entropy — the blocker is architectural, not reward or environment. The next
iteration changes **how the policy is represented and trained**. Full diagnosis,
sources and rationale in `RL_RESEARCH.md`.

- [ ] Phase 0: entropy schedule (decay `ent_coef`) + diagnostics in `train_v3.py`.
- [ ] Phase 1: contextual action masking (`COMPLETE_QUEST`/`LOOT`/cast validity).
- [ ] Phase 2: MoE actor (SoftMoE, ~4 experts) as an SBX `ActorCriticPolicy`.
- [ ] Phase 3: entity-attention encoder over the structured frame.
- [ ] Phase 4: hierarchical options + RND intrinsic bonus.

---

## Status snapshot (2026-08-27)

- PPO + shared memory + MySQL: **done** (v0.1.0).
- Native reward (§2): **done** (v0.2.0, + spellLearned in v0.6.0). Shaping terms are
  diagnostic-only.
- Faithful structured state (§1): **done** (v0.3.0). Packed entity-centric frame over
  shm v2; `MlpPolicy` consumes a flattened projection until DreamerV3.
- Action space v2 (§4): **done** (v0.4.0) — 41 actions: point-nav, entity-index targeting,
  spellbook-index casting, `INTERACT_TARGET`.
- Baseline spells (§6): **done** (v0.6.0) — bots born with level-1 abilities; trainer
  purchases remain the learned path to higher ranks.
- Research + design: **done** — `DESIGN.md` specifies the state schema, action rework,
  and shm v2; `RL_RESEARCH.md` (2026-08-26) diagnoses the max-entropy plateau and
  proposes the MoE/attention/hierarchy v2 policy.
- **Plateau diagnosis (2026-08-26)**: PPO sparse (v13/v14_bc), PPO dense (v15_dense),
  and R2-Dreamer all stall at max entropy (~log 41). `v15_dense`: kills 0.87→0.16,
  `explained_variance` 0.90→0.586, entropy ~3.6 over ~248M steps. Level-band respawn +
  preserve-characters shipped and working; the stall is algorithmic (constant entropy
  bonus + non-stationarity-induced plasticity loss) — see `RL_RESEARCH.md`.
- Next: §10 Phase 0 (entropy schedule) is the cheapest test of the diagnosis.
- **Phase 0 entropy schedule: VERDICT CONFIRMED (2026-08-27).** Fast decay (arm `i3`,
  warm from i2@60M, 5% window) collapsed policy entropy **3.42 → 1.40 nats** (vs max
  log41=3.71) the moment `ent_coef` hit 0.0005, while slow-decay arms stayed pinned at
  3.4–3.5. Kills diverged +8% (i3 0.76 vs i2 0.70) and reward less negative. **The
  constant entropy bonus was the binding constraint** — removing it lets the policy
  specialize. Follow-up: i1 (long lineage) switched to fast decay too, so both fast arms
  now exploit; i2 kept as slow control. §10 Phase 1 (policy architecture) is now
  unblocked.
- **Throughput scaling (2026-08-27)**: rndbots disabled (freed ~1 core), 800 bots = the
  per-instance sweet spot (1600 is worse), **3 instances (3×800) = ~73k bot-steps/s**
  (26.3k + 24.9k + 22.1k), load ~15/16 (box saturated). All training processes at
  `nice +5` so desktop use wins under contention. Multi-instance is the scaling path;
  the 5700X/128GB box is available for more instances. See AGENTS.md
  "Multi-instance scaling".
- Ops hardening (v0.6.0): shutdown crash class fixed (TCP handler disabled), systemd
  `KillMode=process`, KL guard over-braking fixed (`NEURALBOT_KL=0`, lr 1.5e-4).
- Earlier: iter 1–4 trained; kills/loot/xp all climbing; spell learning blocked.
