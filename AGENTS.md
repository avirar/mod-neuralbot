# AGENTS.md — mod-neuralbot

Handover notes for agents working on this project. This is the "current state + how to do
things" summary. The full design/plan live in `README.md`, `DESIGN.md`, `ROADMAP.md`,
`CHANGELOG.md`.

## What this is

`mod-neuralbot` trains reinforcement-learning bots that play **real WoW 3.3.5a** on an
AzerothCore server. Bots are controlled **server-side** (packet injection), stepped in a
batch over shared memory, and trained with SBX PPO (JAX) in Python. The current direction
is a rebuild toward **faithful state + native reward + no hardcoded auto-services**
(see `ROADMAP.md` §0). Coexists with `mod-playerbots` ("true diversity" of AI players).

## Repository layout

Three **independent** nested git repos (no submodules; `modules/` is gitignored by the
parent):

| Repo | Path | `origin` (avirar fork) | `upstream` | Branches |
|------|------|------------------------|------------|----------|
| azerothcore | `/home/luke/GIT/azerothcore-wotlk` | `avirar/azerothcore-wotlk` | `mod-playerbots/azerothcore-wotlk` | `neuralbot` (ours, build branch), `Playerbot` (mirrors upstream) |
| playerbots  | `…/modules/mod-playerbots` | `avirar/mod-playerbots` | `mod-playerbots/mod-playerbots` | `master` (upstream), `neuralbot-bc` (ours: `OnPlayerbotActionExecuted` hook) |
| neuralbot   | `…/modules/mod-neuralbot` | `avirar/mod-neuralbot` | — | `master` |

- Build/run happens from the parent repo on the **`neuralbot`** branch.
- Our parent-repo changes (`conf/dist/config.sh` → `RelWithDebInfo`, a local DB-migration
  tweak) are committed on `neuralbot`. Keep `Playerbot` a clean upstream mirror.
- Sync with `./modules/mod-neuralbot/scripts/sync_upstream.sh` (fetch + merge + push;
  run weekly, **not** while the worldserver is training).

## Commands

```bash
# Build (from parent repo, on the `neuralbot` branch)
cd /home/luke/GIT/azerothcore-wotlk
bash acore.sh compiler build          # incremental
bash acore.sh compiler compile        # fresh (slow)

# Run training (module dir) — preferred detached launch
cd modules/mod-neuralbot
NEURALBOT_LR=1.5e-4 NEURALBOT_ENT=0.01 NEURALBOT_KL=0 \
    scripts/launch_trainer.sh wow_neuralbot_model_v13

# Or directly (foreground)
source .venv/bin/activate
NEURALBOT_TIMESTEPS=20000000 python3 python/train_v3.py
```

- Server: `env/dist/bin/worldserver`; config `env/dist/etc/modules/mod-neuralbot.conf`.
  Start pattern (stdin EOF = shutdown):
  `setsid nohup bash -c 'exec tail -f /dev/null | ./worldserver' >> ws_console.out 2>&1 < /dev/null &`
- Episode stats → MySQL `acore_characters.neuralbot_episodes`.
- Python venv: `modules/mod-neuralbot/.venv` (Python 3.12, `--system-site-packages`).

### QoL scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `start_training.sh [prefix] [steps]` | Start worldserver (if down) + wait for 400 bots + launch/resume trainer; clears `.maintenance` |
| `stop_training.sh [world]` | Stop trainer (optionally + worldserver); sets `.maintenance` |
| `status.sh` | One-shot health snapshot (processes, bots, recorder, trainer metrics, disk) |
| `bc_report.sh [hist\|progress] [--limit N]` | Run `bc_analyze.py` on `python/bc_demos/demos.bin` |
| `archive_bc_demos.sh [--force]` | Move demos.bin to NAS (refuses while recorder is live unless `--force`) |
| `archive_to_nas.sh` | Daily NAS archiver (episodes, old checkpoints, logs, `Server.log`) |
| `launch_trainer.sh <prefix>` | Trainer-only detached launch (assumes worldserver + bots up) |
| `sync_upstream.sh` | Weekly upstream sync (fetch + merge + push) |

- BC demo analysis: `python3 python/bc_analyze.py {hist|progress} python/bc_demos/demos.bin`.
- The `OnPlayerbotActionExecuted` hook lives in the **parent repo** (`ScriptMgr.h` +
  `ScriptDefines/PlayerbotsScript.cpp`, branch `neuralbot`) and is fired from
  **mod-playerbots** `Engine::ListenAndExecute` (branch `neuralbot-bc`). Both are
  required for the BC recorder to fire — build with mod-playerbots on `neuralbot-bc`.
- Trainer log: `ls -t python/logs/train_v13_*.log | head -1` (NOT `tail -1` — picks the stale log).
- Resume lineage: `cp <newest *_steps.zip> wow_neuralbot_model_v13.zip` then relaunch
  (or `start_training.sh` / the watchdog do it automatically).
- `pkill -f "[t]rain_v3.py"` (bracket trick) — never put a literal `train_v3.py` in the
  same shell that runs pkill.

## Environment gotchas

- **GCC 16 + jemalloc.** Bundled `deps/jemalloc` 5.2.1 fails to build with GCC 16
  (`std::__throw_bad_alloc` removed). Fix is one line: `throw std::bad_alloc();` in
  `deps/jemalloc/src/jemalloc_cpp.cpp` — **already fixed in upstream azerothcore**, so
  after syncing you don't need the local patch.
- **Build dir is `var/build/obj`** (not the root `build/`, which is a stale dead tree).
  `acore.sh compiler build` = cmake configure + incremental compile; it re-globs module
  sources each time, so **new `.cpp` files are picked up automatically**.
- **World DB migration backlog.** After the upstream sync the `acore_world` DB is far
  behind (~600 pending `db_world` updates, up to `2026_08_07`). On startup the
  worldserver applies them one-by-one (several minutes) before it listens on 8085; it is
  resumable, so re-running continues where it left off. Don't mistake the quiet console
  for a hang — see logging note.
- **Console logging is filtered to Error level** by default in the local
  `env/dist/etc/worldserver.conf` (`Appender.Console=1,2,0`). Bumped to `1,4,0` (Info)
  on 2026-08-18 so the terminal shows detail; the full log still goes to
  `env/dist/bin/Server.log` (Debug level). `Errors.log` is empty unless there are errors.
- **Performance:** the worldserver world thread (sessions + NeuralBot step loop) is the
  training-throughput gate and is serial by design; `MapUpdate.Threads = 5` in
  `env/dist/etc/worldserver.conf` offloads map updates (~17–21k fps). Next lever if
  needed: stagger BuildFrame's 60-yd grid scans across steps, or a second worldserver
  shard (deferred).
- **No submodules.** `deps/*` and `modules/*` are plain tracked dirs / separate repos.
- **DB:** `acore:abc@127.0.0.1:3306`, databases `acore_auth` / `acore_characters` /
  `acore_world`. MySQL runs on the host (port 3306), not the `ac-database` docker
  service.
- **Git creds:** HTTPS via `~/.git-credentials` (user `avirar`). `gh` CLI token is stale;
  use git, not gh.
- **Watchdog KillMode:** `neuralbot-watchdog.service` must keep `KillMode=process` — the
  default `control-group` made systemd SIGTERM the worldserver/trainer every time the
  watchdog script exited (clean `Halting process…` ~72 s after each boot). Unit tracked
  at `scripts/neuralbot-watchdog.service`; install to `~/.config/systemd/user/`.
- **Storage:** overflow (old logs, generated weights/checkpoints) goes to `~/NAS/temp/neuralbot`.
  Automated daily by `scripts/archive_to_nas.sh` (systemd user timer `neuralbot-archive.timer`,
  04:23): episodes rows >2 d old, checkpoints beyond newest 5, old logs, `Server.log` >500 MB
  (~1 GB/hour while training). Run it manually after big housekeeping.
- **DBC data:** `SkillLineAbility.dbc` (spell → skill-line auto-learn, `AcquireMethod`,
  `RaceMask`/`ClassMask`, `MinSkillLineRank`) lives in `env/dist/bin/dbc/`. Queryable via
  `~/GIT/acore-data` (MCP server; set `ACORE_DBC_PATH` and `ACORE_FORMAT_FILE`).

## Current state (2026-08-23, v0.6.0)

Done:
- PPO + shm IPC + MySQL logging (v0.1.0); native reward (v0.2.0); structured frames
  (v0.3.0).
- v0.3.1: bot-name `GM` suffix fix (reserved names killed one env slot every boot).
- v0.3.2: bots revive on episode end (dead slots used to loop length-1 episodes forever).
- v0.4.0: **action space v2** — 41 actions: MOVE_TO_TARGET (MoveChase nav),
  TARGET_ENTITY_0..17 (frame-index parity via BuildFrame guid cache),
  CAST_SPELL_0..7 (frame spells[] order, passives filtered), INTERACT_TARGET
  (gated 5.5yd context action: quest/trainer/vendor/chest), nearest-enemy/friendly/corpse
  targeting, attack start/stop, COMPLETE_QUEST, LOOT. Episode table has act_0..act_40.
- v0.5.0: pipelined shm protocol (Python harvester thread + depth-3 queue + C++
  backpressure `ObservationsPending`); `MapUpdate.Threads` 3 → 5.
- v0.6.0: **baseline spells** (bots born with ~45 level-1 abilities; `Player::Create`
  learned them as temporary — factory converts `PLAYERSPELL_TEMPORARY→NEW` before save);
  **`spellLearned` is a native reward term**; trainer purchases stay the learned path to
  higher ranks (no auto-maintenance).
- Ops hardening (v0.6.0): shutdown crash class fixed (dead TCP handler disabled,
  `NeuralBot.WebSocketPort=0`); systemd `KillMode=process` (was control-group, which
  SIGTERM'd worldserver/trainer every watchdog pass); KL-guard over-braking fixed.
- Ops: daily NAS archiver (`scripts/archive_to_nas.sh`, timer 04:23) + training watchdog
  (`scripts/watchdog.sh`, `neuralbot-watchdog.service`; `.maintenance` flag in the module
  dir stands it down during manual work).

### Active hyperparameters (v13 lineage)
- `NEURALBOT_LR=1.5e-4`, `NEURALBOT_ENT=0.01`, `NEURALBOT_KL=0` (guard off — it was
  pinning LR at its 1e-5 floor; SBX 0.25.0 *does* serialize `target_kl`/`adaptive_lr`).
- `NEURALBOT_TIMESTEPS=1000000000`, `NEURALBOT_REWARD_MODE=symlog` (default; the old
  `[-1,0.3]` clip collapsed +20/+10/+0.01 onto one value — symlog preserves the gradient).
- gamma 0.999, gae_lambda 0.98, n_steps 1024.
- Watchdog passes ENT/KL and defaults LR to 1.5e-4 in `scripts/watchdog.sh`.

### Critical-review response (2026-08-23)
A code review + RL-literature survey (`REVIEW.md`) established the agent was **not
learning** (~1B steps at max entropy, EV ≈ 0, flat reward/kills — all verified). The
accepted plan:
- **Tier 0 (done, v13)**: per-field observation normalization in `flatten_frames` +
  `symlog` reward. Fresh v13 lineage, v12 retired.
- **Tier 1 (in progress)**: measure-then-map BC warm-start from mod-playerbots +
  a DreamerV3/NE-Dreamer world-model spike in parallel.
- BC recorder live: `NeuralBot.BcRecordPath` (worldserver appends fixed-size
  `NeuralBotBcRecord`s = header + full frame per executed playerbot action);
  `python/bc_analyze.py` reports the action histogram + per-action progress. The
  measurement decides *which* expert behaviors are worth cloning (return-filtered BC
  — only actions that lead to xp/kills/loot get distilled, so playerbot bugs don't
  propagate).

### World-model spike (2026-08-23) — READY to run
- **NE-Dreamer / R2-Dreamer (PyTorch)** chosen over danijar/dreamerv3 (JAX, idiosyncratic
  `embodied` framework). One codebase covers `ne_dreamer`/`r2dreamer`/`dreamer` via
  `model.rep_loss`; NE-Dreamer's temporal transformer targets long-horizon
  memory/navigation (DMLab Rooms) — the walk-to-trainer/quest-giver pain point.
- **Implementation (all in-repo):** `python/neuralbot_shm.py` (dependency-free shm
  client + `flatten_frames`, single source of truth), `python/wow_world_model_env.py`
  (`WoWWorldModelEnv` duck-typing the `ParallelEnv` contract), `scripts/setup_worldmodel_venv.sh`
  (Python 3.11 + `torch==2.8.0+cu128` via uv). r2dreamer is cloned into `worldmodel/r2dreamer`
  (gitignored); the config/hook changes are captured in `worldmodel_wow.patch`
  (`envs/__init__.py` wow branch, `configs/env/wow.yaml`, `configs/configs.yaml`
  buffer 1e6/cpu). See `WORLD_MODEL_SPIKE.md`.
- **Run:** `worldmodel/.venv311/bin/python worldmodel/r2dreamer/train.py env=wow` —
  **requires stopping the PPO trainer first** (both drive the same 400-bot shm; they
  are mutually exclusive). The BC recorder (playerbot-side) keeps running either way.
- Key facts: batched env of 400 is correct (buffer treats each env index as an
  independent trajectory); raw native reward, no clip (WM reward head `symexp_twohot`
  spans symlog ±20); `is_terminal` = death component (reward idx 3) > 0.

Training runs:
- v4 = 16-action baseline (15.5M steps, checkpoints kept) — reward flat at ~0, kills
  0.01/ep.
- v5 = v0.4.0 action space, resumed across rebuilds from `*_steps.zip` checkpoints.
- v8–v12 = variance-reduction stack (lr sweep, KL guard, reward clip). Kill-rate oscillation
  (~0.44↔0.20, ~1h period) traced to the KL guard never actually engaging (SBX serialization
  bug) — now fixed.
- v13 = Tier-0 fixes (normalized obs + symlog reward), fresh lineage, lr 1.5e-4 guard off.

Next (ROADMAP): Tier 1 — BC measurement (record playerbot demos, histogram, decide
what to clone), DreamerV3/NE-Dreamer spike; defer §3 (kill auto-services) until the
warm-start or a denser signal exists.

## Conventions

- Commit style: Conventional Commits (`feat(NeuralBot): …`, `fix(NeuralBot): …`,
  `chore`, `docs`).
- Comments are welcome; document intent (the `// ── native ──` style is fine).
- C++: 4-space indent, no tabs (matches AzerothCore `.editorconfig`).
- The module has no upstream; `origin` (`avirar/mod-neuralbot`) **is** the source of truth.
- Reward/obs schema changes must stay in sync across `NeuralBotFrame.h` (packed wire
  structs, `static_assert` sizes), `NeuralBotMgr.cpp` (frame write), `shared_memory_env.py`
  and `neuralbot_client.py` (numpy dtypes — keep `align=False`, verify `FRAME_BYTES`).
