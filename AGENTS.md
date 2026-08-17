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
- **No submodules.** `deps/*` and `modules/*` are plain tracked dirs / separate repos.
- **DB:** `acore:abc@127.0.0.1:3306`, databases `acore_auth` / `acore_characters` /
  `acore_world`.
- **Git creds:** HTTPS via `~/.git-credentials` (user `avirar`). `gh` CLI token is stale;
  use git, not gh.

## Current state (2026-08-17)

Done:
- PPO + shared-memory IPC + MySQL episode logging (v0.1.0).
- **Native reward** (v0.2.0): `total = XP + gold + level + quest_complete − death`;
  shaping terms are diagnostic-only.
- Research + `DESIGN.md` (faithful state schema, action rework, shm v2).
- Git forks (`avirar/{azerothcore-wotlk,mod-playerbots,mod-neuralbot}`) set up and synced
  to latest upstream.

Next (in order, from `ROADMAP.md`):
1. **§1 Faithful structured state** — replace the 85-float vector with entity-centric
   records over shm (spec in `DESIGN.md`). Touches `NeuralBotCommon.h`,
   `NeuralBotInstance.cpp` (`BuildObservationInto`), `NeuralBotMgr.cpp` (shm write),
   `shared_memory_env.py`, `neuralbot_client.py`.
2. **§3 Kill auto-services** — remove `AutoQuest`, auto-target, auto-loot.
3. **§4 Action rework** — point-nav + entity-index targeting + spellbook-index casting.
4. **§5 DreamerV3** — official `danijar/dreamerv3` (JAX), world model over real state.
5. **§6 Fix spell learning** — bots can't close the 5-yard trainer interaction.

## Conventions

- Commit style: Conventional Commits (`feat(NeuralBot): …`, `fix(NeuralBot): …`,
  `chore`, `docs`).
- Comments are welcome; document intent (the `// ── native ──` style is fine).
- C++: 4-space indent, no tabs (matches AzerothCore `.editorconfig`).
- The module has no upstream; `origin` (`avirar/mod-neuralbot`) **is** the source of truth.
- Reward/obs schema changes must stay in sync across `NeuralBotCommon.h`,
  `NeuralBotMgr.cpp`, `shared_memory_env.py`, and `neuralbot_client.py`.
