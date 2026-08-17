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
| playerbots  | `…/modules/mod-playerbots` | `avirar/mod-playerbots` | `mod-playerbots/mod-playerbots` | `master` |
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

# Run training (module dir)
cd modules/mod-neuralbot
source .venv/bin/activate
NEURALBOT_TIMESTEPS=20000000 python3 python/train_v3.py
```

- Server: `env/dist/bin/worldserver`; config `env/dist/etc/modules/mod-neuralbot.conf`.
- Episode stats → MySQL `acore_characters.neuralbot_episodes`.
- Python venv: `modules/mod-neuralbot/.venv` (Python 3.12, `--system-site-packages`).

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
- **No submodules.** `deps/*` and `modules/*` are plain tracked dirs / separate repos.
- **DB:** `acore:abc@127.0.0.1:3306`, databases `acore_auth` / `acore_characters` /
  `acore_world`. MySQL runs on the host (port 3306), not the `ac-database` docker
  service.
- **Git creds:** HTTPS via `~/.git-credentials` (user `avirar`). `gh` CLI token is stale;
  use git, not gh.
- **Storage:** overflow (old logs, generated weights/checkpoints) goes to `~/NAS/temp/neuralbot`.
  Automated daily by `scripts/archive_to_nas.sh` (systemd user timer `neuralbot-archive.timer`,
  04:23): episodes rows >2 d old, checkpoints beyond newest 5, old logs, `Server.log` >500 MB
  (~1 GB/hour while training). Run it manually after big housekeeping.

## Current state (2026-08-18, v0.4.0)

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
- Ops: daily NAS archiver (`scripts/archive_to_nas.sh`, timer 04:23) + training watchdog
  (`scripts/watchdog.sh`, `neuralbot-watchdog.service`; `/tmp/neuralbot_maintenance` flag
  stands it down during manual work).

Training runs:
- v4 = 16-action baseline (15.5M steps, checkpoints kept) — reward flat at ~0, kills
  0.01/ep.
- v5 = v0.4.0 action space, resumed across rebuilds from `*_steps.zip` checkpoints.

Next (ROADMAP): §3 kill remaining auto-services (AutoQuest, cast auto-target fallback,
COMPLETE_QUEST/LOOT context scans), §5 DreamerV3, curriculum.

## Conventions

- Commit style: Conventional Commits (`feat(NeuralBot): …`, `fix(NeuralBot): …`,
  `chore`, `docs`).
- Comments are welcome; document intent (the `// ── native ──` style is fine).
- C++: 4-space indent, no tabs (matches AzerothCore `.editorconfig`).
- The module has no upstream; `origin` (`avirar/mod-neuralbot`) **is** the source of truth.
- Reward/obs schema changes must stay in sync across `NeuralBotFrame.h` (packed wire
  structs, `static_assert` sizes), `NeuralBotMgr.cpp` (frame write), `shared_memory_env.py`
  and `neuralbot_client.py` (numpy dtypes — keep `align=False`, verify `FRAME_BYTES`).
