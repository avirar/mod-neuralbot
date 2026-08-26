# RL Research — Modern architectures for the NeuralBot stall

Status: research notes + design proposal (2026-08-26). Companion to `REVIEW.md` (the
original critical review) and `ROADMAP.md` (execution order).

## The problem we are solving

Three algorithms (PPO sparse-reward, PPO dense-reward `v15_dense`, R2-Dreamer world
model) × three reward schemes all plateaued the same way: **the policy stays at ~max
entropy and never exploits**. Concretely, `v15_dense` at ~248M steps had
`entropy_loss ≈ −3.6` (≈ log(41), i.e. uniform) while kills *dropped* 0.87 → 0.16/ep and
`explained_variance` fell 0.90 → 0.586. The dense reward is *learnable* (EV hit 0.9
early), so the blocker is **how the policy is represented and trained**, not the reward
or the environment.

## Diagnosis (why we stay at max entropy)

1. **A constant entropy bonus pins the policy near uniform.** SBX PPO's `ent_coef`
   (`NEURALBOT_ENT=0.01`) adds a *maximize-entropy* term every step. With 41 actions that
   is ~0.037/step of gradient pulling toward uniform — the same order as our dense reward
   (damage 0.1–1.0, but only while fighting). The exploration bonus never decays, so the
   policy never has to commit.
   - Literature: PPO's entropy bonus must be **scheduled** (decay) or replaced with an
     intrinsic bonus; a fixed bonus = permanent exploration (Schulman/GRPO "entropy
     rabbit hole" threads, 2025; DAPO/CISPO/DPPO variants all exist to manage entropy).
   - Note the LLM literature studies the *opposite* failure (entropy *collapse* from
     logit diffusion + clipping); our uniform-entropy stall is the classic
     `ent_coef`-too-large / reward-too-weak regime.

2. **Non-stationarity → plasticity loss.** Our environment is violently non-stationary:
   bots level up, the level-band respawn teleports them to new hubs, mobs change, the
   reward distribution shifts. A monolithic MLP loses plasticity (dormant neurons) and
   its value function degrades over time — exactly our `explained_variance` 0.90→0.586.
   - Literature: MoEs measurably reduce dormant neurons and preserve plasticity under
     amplified non-stationarity (Obando-Ceron et al., "Mixture of Experts in a Mixture
     of RL settings", arXiv:2406.18420).

3. **Credit assignment is diluted over 41 actions.** The combat loop is
   `TARGET_NEAREST_ENEMY → ATTACK/CAST`, but probability mass is spread over all 41
   actions by the entropy bonus. The small subset that earns reward gets a weak,
   noisy gradient.

## Key research findings (with sources)

| Finding | Source | Implication for us |
|---|---|---|
| MoEs reduce dormant neurons & preserve plasticity under non-stationarity; help the **actor** more than the critic; experts specialize per task | arXiv:2406.18420 (Obando-Ceron et al., 2024) | Replace the policy MLP with an MoE so combat/exploration/quest experts can specialize instead of overwriting each other |
| SoftMoE (soft differentiable routing) works well under *frequent* task switches; hard top-k routing is brittle | arXiv:2406.18420, Puigcerver et al. 2023 | Use soft routing (our "tasks" = combat vs idle vs quest change constantly) |
| "Big" MoE (each expert = a full network, router only at the head) performs best; per-sample tokenization (one token = one state) | arXiv:2406.18420 | A small number of full expert heads + a router on the flat obs is the simplest effective starting point |
| MENTOR: MoE + task-oriented perturbation for visual RL | Huang et al., ICML 2024 | Perturbation/oracle signals can regularize expert specialization |
| Fixed entropy bonus pins policy near uniform; entropy must be scheduled or replaced by intrinsic reward | Schulman/Zhang/Ciuca threads 2025; spinningup PPO | Decay `ent_coef` or swap to RND intrinsic bonus |
| RND (prediction-error intrinsic reward) gives exploration with minimal overhead, no entropy bonus needed | Burda et al., arXiv:1810.12894 | Replace the entropy bonus with RND so exploration targets *novel states*, not uniform action noise |
| Entity/relational attention over structured observations for RL | arXiv:2206.02855 (entity transformer) | Our `NeuralBotFrame` is already structured (self/target/entities[64]/spells/quests) — attend to the relevant record instead of a flat 1148-vector |
| Hierarchical RL / options decomposes long-horizon tasks into skills | arXiv:2505.12109 (wargaming hierarchical RL) etc. | High-level option policy (explore/combat/quest/loot) + per-skill low-level policies; scaffolding at the *architecture* level, not reward shaping |

## Proposed architecture (v2 policy)

1. **MoE policy (actor).** Replace the flat MLP actor with `router → k experts → head`.
   - Experts: 4 full MLP experts (explore / combat / quest / loot), soft-routed on the
     obs embedding. Router softmax over experts; combine with the soft weights.
   - Keep the critic a plain MLP first (MoE helped the actor more than the critic).
   - Expected effect: combat gradients stop being overwritten by exploration/idle
     gradients — the combat expert can specialize without plasticity loss.

2. **Entity attention encoder.** Feed the *structured* frame (self, target, top-N
   entities, spells, quests) through a small transformer/attention stack instead of the
   1148-float flattened projection.
   - Heads can specialize (enemy head attends to hostile entities, quest head attends
     to `questState`), which is the "attention heads" the user described.
   - Practical step: SBX supports `MultiInputPolicy`; alternatively a custom
     `ActorCriticPolicy` with a torch transformer encoder.

3. **Entropy schedule (or RND).** Decay `ent_coef` 0.01 → 0.0005 across the run (linear
   or exponential), or replace it with an RND intrinsic bonus on the obs embedding so
   exploration is state-novelty-driven, not uniform-noise-driven.
   - This is the *cheapest* lever and can be tried immediately in `train_v3.py`.

4. **Hierarchical skills (options).** A high-level policy picks a skill per N steps;
   each skill is a low-level policy over the same 41 actions (or a per-skill action
   mask).
   - Skills: `explore` (move/target), `combat` (target+attack+cast), `quest`
     (interact/complete-quest/nav-to-QG), `loot`.
   - This is *architectural scaffolding*: it constrains which actions are available per
     skill (a form of action masking), rather than shaping the reward.

## Concrete implementation plan (phased, smallest first)

**Phase 0 — entropy schedule + diagnostics (train_v3.py only, no C++):**
- Decay `NEURALBOT_ENT` over training; log entropy, dormant-neuron fraction, and
  per-action selection frequency so we can *see* whether the policy starts exploiting.
- Sweep `ent_coef` final value (0.01 → 0.0005) and schedule type.
- Expected outcome: policy entropy falls, kills/reward rise — if so, the stall was the
  entropy bonus, not the architecture.

**Phase 1 — action masking / action-space pruning:**
- Mask contextually-invalid actions (e.g. `COMPLETE_QUEST` when no complete quest,
  `LOOT` when no corpse) so the policy only faces valid choices. Reduces the effective
  action space and the entropy-bonus dilution.

**Phase 2 — MoE actor (custom policy):**
- Implement a SoftMoE actor (4 experts, soft routing) as an SBX `ActorCriticPolicy`
  subclass (or a small torch module). Keep critic MLP.
- Compare vs Phase-0 baseline on kills/reward/EV.

**Phase 3 — entity attention encoder:**
- Transformer encoder over the structured frame records; evaluate alone and with MoE.

**Phase 4 — hierarchical options + RND:**
- High-level option policy + per-skill low-level policies; RND intrinsic bonus replaces
  the entropy bonus for exploration.

Each phase is independently buildable and measurable against the current `v15_dense`
baseline (kills ≈ 0.16/ep, EV ≈ 0.586, entropy ≈ 3.6).

## Open questions
- Does the entropy schedule alone break the stall (i.e. was it *just* `ent_coef`)? Phase 0 answers this cheaply.
- SoftMoE vs hard top-k routing for a single evolving MDP (our case) — the paper's soft
  routing wins under frequent *task switches*; ours is one task with drifting dynamics.
- How much of the frame to attend over (64 entities is a lot; top-K by distance?).
- RND prediction target (obs embedding vs next-obs latent) and its interaction with symlog reward.

## Sources
- arXiv:2406.18420 — Mixture of Experts in a Mixture of RL settings (non-stationarity, plasticity, routing).
- MENTOR — MoE + task-oriented perturbation for visual RL (ICML 2024).
- arXiv:1810.12894 — RND (Exploration by Random Network Distillation).
- arXiv:2206.02855 — entity/relational attention for RL.
- arXiv:2505.12109 — hierarchical RL / options (wargaming).
- Schulman/Zhang/Ciuca 2025 threads + spinningup PPO — entropy dynamics & scheduling.
