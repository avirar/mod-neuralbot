# mod-neuralbot — Critical Review & Recommendations

**Date:** 2026-08-23
**Scope:** Code review of `modules/mod-neuralbot`, analysis of live training data (`acore_characters.neuralbot_episodes`), and a survey of the current reinforcement-learning literature and prior work on RL agents playing RPG/MMORPG computer games.

---

## 1. Summary

`mod-neuralbot` trains reinforcement-learning bots that play **real WoW 3.3.5a** on an
AzerothCore server. Bots are controlled **server-side** (packet injection, no game
client), stepped in a batch over shared memory (`/dev/shm`), and trained with SBX PPO
(JAX) in Python. It is an ambitious and genuinely novel project — the "server-side moat"
(the ability to read the complete true game state and run hundreds of parallel bots in
the real game) is a real differentiator that no vision or simplified-sim approach has.

**Verdict:** the engineering is strong and well-documented, but **the agent is not
learning**. After ~1 billion environment steps it has settled at maximum entropy
(uniform-random policy, ~zero explained variance). The approach — pure model-free PPO
with a sparse "native" reward on the full open world — sits in the exact failure regime
documented across the RL literature, and the current scaffolding (auto-services,
curriculum teleport, reward clipping) both masks and degrades the signal.

---

## 2. What is genuinely good (engineering)

- **Clean, well-documented codebase.** `DESIGN.md`, `ROADMAP.md`, `CHANGELOG.md`,
  `AGENTS.md` are unusually thorough; Conventional Commits are followed.
- **Robust IPC.** Packed `#pragma pack(1)` frame (5909 B), versioned shared-memory
  control block, `static_assert` + numpy `dtype.itemsize` schema checks
  (`src/NeuralBotFrame.h`, `python/shared_memory_env.py`).
- **Pipelined throughput design.** Harvester thread + depth-3 queue + C++
  backpressure (`NeuralBotSharedMem::ObservationsPending`) so C++ and Python overlap
  instead of summing latencies.
- **Real debugging discipline.** Correctly diagnosed and fixed a string of non-obvious
  bugs: the reserved `GM`-suffix name bug (`NeuralBotFactory::GenerateBotName`), the
  dead-bot length-1 episode loop (`ReviveIfDead`), the SBX `target_kl` serialization bug,
  the systemd `KillMode=control-group` watchdog kill, the GCC 16 + jemalloc break, and
  the TCP accept-thread shutdown SIGABRT.

---

## 3. The critical problem: it is not learning

Evidence from `acore_characters.neuralbot_episodes` (~1.06M rows, episode IDs 0–670230)
and the live training logs (`python/logs/train_v12_*.log`):

| Metric | Value | Interpretation |
|---|---|---|
| `entropy_loss` | **3.69 ≈ ln(41)** | Policy is essentially **uniform random** over the 41 actions |
| `explained_variance` | **0.003–0.02** | Value function predicts **nothing** |
| `avg_reward` (episode buckets 500k→650k) | 0.007 → 0.010 | flat, no improvement |
| `avg_kills` | 0.169 → 0.220 | flat |
| `avg_xp` | 0.009 → 0.013 | flat |
| `avg_length` | 1483–1492 | episodes terminate at the **1500-step idle timeout**, not from progress |
| action distribution | ~28–44 of each (of ~36 expected uniform) | no action preference learned |

The model has consumed ~1 billion environment steps and reached maximum entropy. A
uniform-random policy would look identical to what is observed. That is the classic
sparse-reward, model-free, long-horizon stall.

---

## 4. Root causes (critical issues)

1. **Wrong algorithm for the regime.** Pure PPO with a sparse, delayed "native" reward on
   an open-world RPG is the exact failure mode documented across the literature (NetHack,
   Minecraft — see §6). The project's own roadmap defers DreamerV3 (§5) behind other
   work while the current PPO loop stalls.

2. **Reward clipping destroys the sparse signal.** `ComputeReward` returns real
   magnitudes (`questCompleted` +20, `spellLearned` +10, `deathPenalty` 10), but Python
   clips to `[-1, 0.3]` (`NEURALBOT_REWARD_CLIP=0.3` in `train_v3.py`). The model
   effectively sees `{0, +0.3, −1}` — a coarse ternary signal with most steps exactly 0.
   This is not "native reward as delivered."

3. **The "faithful, no hacks" claim is not yet true.** `AutoQuest=1` (auto-accept/
   auto-complete), `ATTACK_START`/`CAST_SPELL` auto-target the nearest hostile, and
   `LOOT`/`COMPLETE_QUEST` do context scans. ROADMAP §3 explicitly lists these for
   removal — the current combat behavior is scaffolded, not learned.

4. **Curriculum staging teleports bots next to mobs** (`StageEpisodeStart`). This
   collapses the ~190-step approach phase but in doing so masks the navigation problem
   the project claims to want the agent to learn. It is reward-engineering by environment
   manipulation, in tension with the "faithful environment" principle.

5. **Weak policy architecture for structured state.** The rich entity/spell/quest frame
   is flattened into **1148 unnormalized floats** (health ~hundreds, money in copper
   ~thousands, positions ~thousands, spell IDs ~5 digits) fed to an MLP
   (`shared_memory_env.py::flatten_frames`). "No normalization" is stated as a virtue but
   is a liability for a vanilla MLP. The distance-sorted `TARGET_ENTITY_i` index coupling
   is also unstable frame-to-frame.

6. **Wasteful diagnostics.** `ComputeReward` runs 60 yd/40 yd grid scans + 25-slot quest
   scans *every step* for terms that are "diagnostic-only," taxed across 400 bots.

7. **Housekeeping.** 13 GB `.venv`, 396 model zips (2.9 GB), and hundreds of logs sit in
   the module directory (an archiver exists but the tree is heavy).

---

## 5. Prior art directly relevant to this project

- **[mod-playerbots](https://github.com/liyunfan1223/mod-playerbots)** — the scripted
  (non-learned) bot module this repo already depends on. Its AI is an **expert scripted
  policy** that plays WoW far better than the current learned policy. It is the natural
  source of demonstration data for behavior cloning / imitation warm-start (see §7).
- **[mrdmnd/portunus](https://github.com/mrdmnd/portunus)** — a WoW AlphaZero/MCTS
  tool (WIP) for mythic+ speedruns, using an in-game LUA addon for state exfiltration.
- **[DominikLindorfer/Tensor-WoW](https://github.com/DominikLindorfer/Tensor-WoW)** —
  a pixel-based CNN "rotation bot" for WoW tanks (vision pipeline, not RL policy
  learning of the full game).
- **[ber84130/wow-ai-complete](https://github.com/ber84130/wow-ai-complete)** — a WoW
  automation bot with a "RL-ready" (Dueling DQN) architecture; mostly vision/automation.
- **Aalborg University (2012) — [fuzzy-logic + RL WoW agent](https://projekter.aau.dk/projekter/files/63661656/report.pdf)** — a
  behavior-based + fuzzy-rule agent whose rule *weights* are tuned by RL (online reward
  models). Notable as the rare academic case where RL was used in WoW, and it used RL to
  tune an expert system, **not** to learn a policy from scratch.

None of these trains a from-scratch neural policy that levels a WoW character against the
native game reward — which is precisely mod-neuralbot's goal and also precisely why it is
hard.

---

## 6. Literature survey: latest RL methods & successful RPG agents

### 6.1 Latest methods (2025–2026): world models dominate

Model-based RL with learned **world models** is now the default for long-horizon, sparse-
reward, visually/symbolically complex domains:

- **[DreamerV3](https://arxiv.org/abs/2301.04104)** (Hafner et al., 2023) — *Mastering
  Diverse Domains through World Models*. First algorithm to collect diamonds in Minecraft
  from scratch (single GPU, no human data, ~100M steps). This is the baseline the
  project's ROADMAP §5 already names.
- **STORM** (Zhang et al., 2023) — transformer-based world model outperforming DreamerV3
  on Atari100k.
- **DIAMOND** (Alonso et al., 2024) — diffusion-based world model.
- **[R2-Dreamer](https://arxiv.org/abs/2603.18202)** (2026) — decoder-free world model
  with a redundancy-reduction (Barlow Twins) objective; 1.59× faster than DreamerV3.
- **[NE-Dreamer](https://arxiv.org/abs/2603.02765)** (2026) — next-embedding prediction
  with a temporal transformer; strong on long-horizon memory/navigation (DMLab Rooms).
- **[Optimistic World Models](https://arxiv.org/html/2602.10044)** (2026) — plug-in
  optimistic dynamics loss (O-DreamerV3 / O-STORM) specifically for **sparse-reward**
  environments.
- **[DreamerV3-XP](https://arxiv.org/abs/2510.21418)** (2025) — prioritized replay +
  intrinsic reward (latent reward disagreement) for sparse-reward settings.

The consistent message: for sparse reward + long horizon + rich state, a **learned world
model** (which mod-neuralbot's server-side state is uniquely well-suited to feed) is the
right tool, not vanilla PPO.

### 6.2 Has anyone made an RL model play an RPG computer game? Yes — but never with pure model-free PPO from scratch

- **Diablo I — [DevilutionX-AI](https://github.com/rouming/DevilutionX-AI) and
  [AlphaDiablo/DiabloGym](https://github.com/Diabolically-Handsome/AlphaDiablo)** (the
  closest analog to WoW). Success came from: a **headless deterministic engine
  (~13,000× realtime)**, **macro-actions** (engage/explore/descend/drink/pickup),
  **per-hit + AC-gain reward shaping**, **action masking**, and **imitation-learning
  warm-start from a scripted bot** before PPO. The DiabloGym ledger documents each
  failure mode eliminated per iteration (mean kills 7.6 → 35.2 over 14 versions).
- **Minecraft — [DreamerV3](https://arxiv.org/abs/2301.04104)**: world model + imagined
  rollouts, no human data, collects diamonds. Prior work (VPT, MineRL) needed human
  demonstrations / curricula.
- **NetHack — [NLE](https://github.com/NetHack-LE/nle) and the
  [NeurIPS 2021 Challenge report](https://proceedings.mlr.press/v176/hambro22a/hambro22a.pdf)**:
  the canonical lesson — **symbolic/hybrid agents beat pure neural RL ~4×**, no agent
  came close to "winning," and the best neural results use **behavior cloning warm-start
  + APPO** ([NetHack is Hard to Hack](https://proceedings.neurips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf),
  [LuckyMera](https://arxiv.org/html/2307.08532)).

### 6.3 The consistent pattern

Across every successful RPG/roguelike RL result, the winning ingredients are some
combination of:

1. **Hierarchical / macro actions** (abstract away ~100-step primitive sequences).
2. **Imitation-learning warm-start** from an expert/symbolic/scripted policy.
3. **A learned world model** for imagined long-horizon credit assignment.
4. **Reward shaping or denser auxiliary signals** (not raw sparse reward alone).

All four are **absent** from mod-neuralbot's current loop — it runs raw PPO + native
sparse reward + primitive 41-action space + no demonstrations. The stall is therefore
fully expected, not a bug.

---

## 7. Recommendations (priority order)

1. **Warm-start from `mod-playerbots` (highest leverage).** `mod-playerbots` is already
   in this repo and is a scripted expert policy. Record a few hundred thousand
   `(observation → action)` episodes from it and **behavior-clone** before PPO
   fine-tuning. This is the exact trick that unblocked Diablo and NetHack, and it would
   break the max-entropy stall immediately.

2. **Move DreamerV3 to P0** (ROADMAP §5, currently P1). The literature (DreamerV3,
   STORM, DIAMOND, R2-Dreamer, NE-Dreamer, OWM) says a world model is the right tool for
   sparse reward + long horizon. mod-neuralbot's server-side state gives it *perfect*
   dynamics inputs (better than pixel-based Minecraft), so the "world model learns real
   game dynamics" differentiator is achievable.

3. **Fix the reward before drawing conclusions.** Replace hard clipping
   (`reward_clip=0.3`) with a bounded rescaling, or add a dense *diagnostic-only* proxy
   reward temporarily to verify that credit assignment works end-to-end before trusting
   the native sparse signal.

4. **Kill the auto-services and curriculum teleport** (ROADMAP §3, `NeuralBot.AutoQuest`,
   auto-target fallbacks, `StageEpisodeStart`) so that any measured progress is real
   rather than scaffolded.

5. **Replace the MLP-over-flattened-vector** with a per-entity attention / transformer
   head, normalizing each field type (health, money, positions, spell IDs) into a sane
   range.

6. **Trim diagnostics + housekeeping.** Move diagnostic scans off the per-step hot path;
   archive/delete the 396 model zips and 13 GB `.venv` from the working tree.

---

## 8. References

- SBX PPO (JAX): <https://github.com/araffin/sbx>
- DreamerV3 paper: <https://arxiv.org/abs/2301.04104>
- R2-Dreamer: <https://arxiv.org/abs/2603.18202>
- NE-Dreamer: <https://arxiv.org/abs/2603.02765>
- Optimistic World Models: <https://arxiv.org/html/2602.10044>
- DreamerV3-XP: <https://arxiv.org/abs/2510.21418>
- Diablo I — DevilutionX-AI: <https://github.com/rouming/DevilutionX-AI>
- Diablo I — AlphaDiablo / DiabloGym: <https://github.com/Diabolically-Handsome/AlphaDiablo>
- NetHack Learning Environment: <https://github.com/NetHack-LE/nle>
- NeurIPS 2021 NetHack Challenge report: <https://proceedings.mlr.press/v176/hambro22a/hambro22a.pdf>
- NetHack is Hard to Hack (BC + APPO): <https://proceedings.neurips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf>
- LuckyMera (hybrid NetHack agent): <https://arxiv.org/html/2307.08532>
- mod-playerbots: <https://github.com/liyunfan1223/mod-playerbots>
- Portunus (WoW AlphaZero, WIP): <https://github.com/mrdmnd/portunus>
- Tensor-WoW (pixel CNN rotation bot): <https://github.com/DominikLindorfer/Tensor-WoW>
- Aalborg fuzzy-logic + RL WoW agent (2012): <https://projekter.aau.dk/projekter/files/63661656/report.pdf>
