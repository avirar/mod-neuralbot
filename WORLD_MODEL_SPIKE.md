# World-Model Spike — Integration Plan (R2-Dreamer / NE-Dreamer)

**Date:** 2026-08-23 · **Status:** research only (no code changes)
**Goal:** wrap the batched shared-memory env (400 bots, one batched step, Discrete(41),
1148-dim normalized flat obs or raw 5909-byte frame) into the R2-Dreamer / NE-Dreamer
training loop, and define a first spike config.

Repos examined (shallow clones in `/tmp/opencode`):
- R2-Dreamer — <https://github.com/NM512/r2dreamer>
- NE-Dreamer — <https://github.com/corl-team/nedreamer>

NE-Dreamer is a fork of R2-Dreamer: identical env contract, identical `envs/parallel.py`,
`buffer.py`, and `MultiEncoder` (both support non-image "proprio" observations via an
`mlp_keys` regex). The only material difference is the representation loss
(`rep_loss: "ne_dreamer"` adds a temporal transformer that predicts the next encoder
embedding; R2-Dreamer uses a same-timestep Barlow-Twins loss).

---

## 1. The env contract (what our adapter must implement)

The trainer/dreamer never touch `gymnasium` step/return directly; they go through a
`ParallelEnv`-shaped object. The **only** interface the training loop depends on is:

| Member | Expected | Source |
|---|---|---|
| `.env_num` | int (number of envs) | `r2dreamer/envs/parallel.py:30` |
| `.observation_space` | `gym.spaces.Dict` | `envs/parallel.py:22` |
| `.action_space` | `gym.spaces.Box` with `.discrete=True` attr, or `Discrete` | `envs/wrappers.py:48` |
| `.step(action, done)` | `(TensorDict, done_tensor)` | `envs/parallel.py:40` |

`step(action, done)` semantics (from `r2dreamer/envs/parallel.py:40-77`):
- `action` shape `(B, A)` (float, any device); `done` shape `(B,)` bool.
- For each env `i`: if `done[i]` → reset it and return its fresh obs with **reward 0,
  done False**; else apply `action[i]` and return `(obs, reward, done, info)`.
- Returns a **`torchrl` `TensorDict`** with `batch_size=(env_num,)`, keys = the stacked
  observation dict fields **plus** `"reward"` (float32), plus a separate `(B,)` bool
  `done` tensor. 1-D fields are unsqueezed to `(B,1)` (`lift_dim`, `parallel.py:34`).

Observation dict keys the model requires (from `r2dreamer/envs/dmc.py:62-89`,
`dreamer.py`):
- `"is_first"` — True only on the first step after a reset (resets RSSM state).
- `"is_last"` — True on episode end (truncation **or** terminal).
- `"is_terminal"` — True only on a *true* terminal (drives `cont = 1 - is_terminal`,
  `dreamer.py:431`).
- one or more payload keys (e.g. `"image"` for vision, or our `"obs"` vector).
- `"reward"` is added by `ParallelEnv`, not part of the observation space.

Action contract for **discrete** spaces (from `envs/wrappers.py:48-74` `OneHotAction`
and `dreamer.py:30,45-52`):
- `make_env` wraps a `Discrete(n)` env in `OneHotAction`, which replaces the action
  space with `Box(0,1,shape=(n,))` and sets `space.discrete = True`.
- `Dreamer.__init__` then reads `act_dim = act_space.n if hasattr(act_space,"n") else
  sum(act_space.shape)` → for a Box that's `41`; `hasattr(act_space,"discrete")` selects
  the `onehot` actor distribution.
- The trainer passes the actor's **one-hot** output `(B,41)` straight to
  `envs.step(act, done)`; the env is responsible for `argmax` → integer index.

Reward contract: the world-model reward head is `symexp_twohot` with bins spanning
symlog-space `[-20, +20]` (`distributions.py:242-251`). Our native reward terms
(death −10, quest +20, spell +10, gold/xp small) fit this range **without clipping** —
the WM path does not need the `[-1, 0.3]` hard clip that crippled the PPO path.

---

## 2. (a) Minimal env adapter, method by method

Do **not** subclass or route through `ParallelEnv`/`Parallel` (that spawns one
subprocess per env; see §3). Write one `WoWWorldModelEnv` that duck-types the interface
above and talks to `/dev/shm/neuralbot_shm` synchronously (reuse the mmap/layout code
from `python/shared_memory_env.py`; drop the async reader-thread pipeline for the spike,
because `is_first`/`is_terminal` derivation needs strict obs↔action alignment).

```python
class WoWWorldModelEnv:
    def __init__(self, num_bots=400, obs_size=1148, action_size=41):
        self.env_num = num_bots
        self.observation_space = gym.spaces.Dict({
            "obs":         gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32),
            "is_first":    gym.spaces.Box(0, 1, (1,), np.float32),   # excluded by encoder
            "is_last":     gym.spaces.Box(0, 1, (1,), np.float32),
            "is_terminal": gym.spaces.Box(0, 1, (1,), np.float32),
        })
        act = gym.spaces.Box(0, 1, (action_size,), np.float32)
        act.discrete = True                 # one-hot action; see §1
        self.action_space = act

        self._shm = ShmClient(num_bots)     # mmap /dev/shm/neuralbot_shm (sync)
        self._is_first = np.ones(num_bots, dtype=bool)

    def step(self, action, done):
        # action: (B, A) one-hot float tensor; done: (B,) bool tensor
        act = action.detach().cpu()
        done_np = done.cpu().numpy() if torch.is_tensor(done) else np.asarray(done)
        idx = act.argmax(dim=-1).numpy().astype(np.uint8)
        idx[done_np] = 0                     # NOOP for envs being reset
        obs_flat, rewards, dones, comp = self._shm.step(idx)  # write actions, read frames

        is_first    = self._is_first.astype(np.float32)
        is_terminal = (comp[:, 3] > 0).astype(np.float32)      # death component (index 3)
        is_last     = dones.astype(np.float32)
        self._is_first = dones                # envs that just ended are "first" next step

        td = TensorDict({
            "obs":         torch.as_tensor(obs_flat, dtype=torch.float32),  # (B,1148)
            "is_first":    torch.as_tensor(is_first),
            "is_last":     torch.as_tensor(is_last),
            "is_terminal": torch.as_tensor(is_terminal),
            "reward":      torch.as_tensor(rewards, dtype=torch.float32),
        }, batch_size=(self.env_num,))
        td = {k: v.unsqueeze(-1) if v.ndim == 1 else v for k, v in td.items()}  # lift_dim
        return TensorDict(td, batch_size=(self.env_num,)), torch.as_tensor(dones)
```

Notes:
- `comp[:, 3]` is `deathPenalty` (`WriteFrameReward` component order, mirrored in
  `python/neuralbot_client.py` `REWARD_COMPONENT_KEYS`: index 3 = `"death"`).
- `obs_flat` is the **normalized** 1148 projection (Tier 0), produced by reusing
  `flatten_frames` from `shared_memory_env.py`.
- Rewards should be the **raw native total** (no `[-1,0.3]` clip); `symexp_twohot`
  handles the magnitude.
- If the raw 5909-byte frame is preferred later (world model consumes structured
  records directly), expose it as additional keys (e.g. `"entities"`, `"spells"`,
  `"quests"`) and widen `mlp_keys` — the `MultiEncoder` already handles arbitrary
  per-key shapes (`networks.py:99-141`).

`is_first`/`is_last`/`is_terminal` semantics mapped onto our shm `done`:
- `is_terminal[i] = 1` iff the bot **died** this step (`deathPenalty` component > 0).
- `is_last[i] = done[i]` (death, idle-timeout, or max-steps all end the episode).
- `is_first[i] = 1` iff the bot was reset on the prior step (C++ already calls
  `ResetRewardTracking` + `ReviveIfDead` + `StageEpisodeStart` inside
  `ProcessSharedMemoryStep` when a frame is done, so the next frame *is* the fresh
  episode start — no separate reset primitive is needed).

---

## 3. (b) Replay-buffer treatment of parallel envs — one batched env of 400 is correct

- `r2dreamer/buffer.py` uses a torchrl `ReplayBuffer` with `LazyTensorStorage(max_size,
  ndim=2)` and `SliceSampler(num_slices=batch_size, traj_key="episode",
  strict_length=True)` (`buffer.py:14-21`). `add_transition` stores one batched
  `(B, 1, ...)` row per step (`buffer.py:23-26`).
- The trainer sets `trans["episode"] = episode_ids` where
  `episode_ids = torch.arange(env_num)` is **constant per env index**
  (`trainer.py:116-118,164`). The buffer therefore treats each of the 400 rows as an
  **independent trajectory stream** (the RSSM state reset is handled by `is_first`, not
  by ending the trajectory). This is exactly our 400 independent bots.
- Consequence: **a single batched env of 400 is the right choice.** The `ParallelEnv`
  (one subprocess per env) is the wrong abstraction for us because our C++
  `ProcessSharedMemoryStep` steps all 400 bots in one call — you cannot independently
  step bot #3 without stepping bot #7. The trainer never inspects `ParallelEnv`'s
  internals; it only needs the `step(action, done)`/`env_num`/spaces interface, so a
  batched env is a drop-in.
- `strict_length=True` requires each trajectory to have ≥ `batch_length+1` (=65)
  contiguous samples. Our episodes are ~1500 steps, so this holds; note the trainer
  keeps `episode_ids` constant so short episodes are still sampable (`trainer.py:117`).
- Buffer capacity: with 400 envs, `max_size=5e5` (default) holds only 1250 steps of
  history (~1 episode); bump to `1e6` (2500 steps). Memory: one row ≈
  obs 1.8 MB + stoch 0.8 MB + deter 3.3 MB ≈ 6 MB ⇒ 1e6 transitions ≈ 1250 rows ≈
  **7–8 GB**. Keep `storage_device: 'cpu'` (NE-Dreamer already defaults to this;
  R2-Dreamer defaults to `${device}`).

---

## 4. (c) Python / torch / CUDA concerns (RTX 5090 + current stack)

| Item | Requirement | Our environment |
|---|---|---|
| Python | **3.11** (`r2dreamer/pyproject.toml`: `requires-python = ">=3.11,<3.12"`; NE-Dreamer README "tested with Python 3.11") | Only **3.12.13** installed. **Install Python 3.11** (e.g. `uv python install 3.11`) and create a **separate venv** — do not pollute the neuralbot JAX/SBX venv. |
| torch | **2.8.0** (both repos pin it) | RTX 5090 is Blackwell `sm_120`; the default `cu126` wheel does **not** support it. Install the CUDA 12.8 build: `torch==2.8.0+cu128` from `https://download.pytorch.org/whl/cu128`. |
| torchrl / tensordict | 0.9.2 / 0.9.1 (paired with torch 2.8) | not installed — add to the new venv. |
| gym | R2-Dreamer: `gymnasium==1.2.0`; NE-Dreamer: `gym==0.23.1` | Only needed for the space *objects* in the adapter; prefer R2-Dreamer's newer `gymnasium`. |
| Driver / CUDA | needs CUDA ≥ 12.8 runtime | Driver **610.57.04**, toolkit **13.3** — compatible with cu128 wheels. |
| numpy | R2-Dreamer `1.26.0` | fine (independent venv). |

Recommendation: **start with R2-Dreamer** (its README notes a ~5× faster DreamerV3
repro, and R2-Dreamer adds ~1.6× on top; cleaner `pyproject`/gymnasium). Treat NE-Dreamer
as the second stage — same adapter, just `rep_loss: "ne_dreamer"` — and it is the
differentiated bet for our long-horizon/credit-assignment problem.

---

## 5. (d) Recommended first spike config

| Config key | Value | Rationale |
|---|---|---|
| `env_num` | **400** | must match the shm batch; each bot = one trajectory |
| `action_repeat` | **1** | one shm round = one step |
| model | **`size12M`** (`deter: 2048, hidden: 256, discrete: 16, units: 256`) | matches the NE-Dreamer paper's 12 M-param agent; drop to `size25M` only if needed |
| obs key | `"obs"` (1148 normalized), `mlp_keys: 'obs'`, `cnn_keys: '$^'` | mirrors `configs/env/dmc_proprio.yaml` (`mlp_keys: '.*'`, `cnn_keys: '$^'`) |
| `rep_loss` | `"r2dreamer"` (then `"ne_dreamer"`) | decoder-free; no image reconstruction needed for a flat-vector obs |
| `batch_size` / `batch_length` | 16 / 64 (defaults) | drop `batch_length` to 32 if OOM |
| `train_ratio` | **64** (default 512; DMLab uses 32) | lower it so the 400-env data rate doesn't swamp the GPU |
| `buffer.max_size` | **1e6**, `storage_device: 'cpu'` | see §3 memory math |
| `steps` (first checkpoint) | **2e6 env steps** | enough to see `dyn_entropy`/`rep_entropy`/`reward` loss/`episode/score` move; ~10–20 min wall-clock at train_ratio 64 on a 5090 |
| `eval_episode_num` | **0** | no second worldserver; monitor `episode/score` + `episode/length` already logged from the train envs (`trainer.py:128-138`) |
| reward | raw native, **no clip** | `symexp_twohot` covers ±20 |
| `horizon` / `imag_horizon` | 333 / 15 (defaults) | lever: our episodes are ~1500 steps, so a longer `horizon` may help later |

Judging the spike: success = `train/dyn_entropy` and `train/rep_entropy` **fall below
`log(K)`** (the RSSM starts learning structure), the reward-head loss drops, and
`episode/score` (native return) **trends up** — i.e. entropy decreases and explained
reward rises, the exact signals the PPO run never produced.

---

## 6. Repo-side hooks required later (not now)

1. `r2dreamer/envs/__init__.py::make_envs` — add a `suite == "wow"` branch returning
   `(WoWWorldModelEnv(...), None, obs_space, act_space)`.
2. `configs/env/wow.yaml` — new env config (values from §5).
3. `configs/configs.yaml` — default env `wow`, model `size12M`.
4. Reuse `python/neuralbot_client.py` dtypes + `shared_memory_env.py::flatten_frames`
   (import the module dir into the new venv's path) to avoid a second copy of the wire
   layout.

## 7. Risks / caveats

- **Curriculum teleport creates state discontinuities.** `StageEpisodeStart` teleports
  bots next to mobs at episode start; the world model must learn this as a hard jump.
  It makes dynamics *harder* to learn — another argument for removing it once a denser
  signal exists (consistent with the earlier review).
- **`is_terminal` vs `is_last` are conflated at the C++ level.** Every `done` triggers a
  reset (revive+stage), so "true terminal" (death) and "truncation" (idle) both reset.
  The adapter distinguishes them via the death component, but the world model still sees
  a reset on both. Acceptable for a spike; flag for later.
- **Python 3.11 is not installed.** Blocking prerequisite for the spike environment.
- **Synchronous shm step** (no reader-thread pipeline) will be slightly slower than the
  PPO path but is required for correct `is_first`/`is_terminal` bookkeeping; at 400
  envs this is not the bottleneck (GPU is).
