"""
WoWWorldModelEnv — batched shared-memory env duck-typing the R2-Dreamer / NE-Dreamer
`ParallelEnv` interface (see WORLD_MODEL_SPIKE.md).

The trainer only ever touches `.env_num`, `.observation_space`, `.action_space` and
`.step(action, done)`; it never inspects the env's internals. One batched env of 400 is
the correct abstraction because the C++ `ProcessSharedMemoryStep` steps all 400 bots in
a single call (you cannot step bot #3 without stepping bot #7) — `ParallelEnv`'s
subprocess-per-env model is wrong here.

Rewards are the RAW native total (no `[-1, 0.3]` clip, no symlog): the world-model
reward head is `symexp_twohot` spanning symlog ±20, which covers our native magnitudes
(quest +20, spell +10, death −10) directly.
"""
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(__file__))

from neuralbot_client import ACTION_COUNT, NUM_BOTS, REWARD_COMPONENT_KEYS  # noqa: E402
from neuralbot_shm import ShmClient, OBS_FLAT_SIZE  # noqa: E402

# Reward-component index of "death" (drives is_terminal). Mirrors WriteFrameReward order.
DEATH_COMPONENT = REWARD_COMPONENT_KEYS.index("death")  # 3


class WoWWorldModelEnv:
    """Duck-types the R2-Dreamer/NE-Dreamer ParallelEnv contract over /dev/shm."""

    def __init__(self, num_bots: int = NUM_BOTS, obs_size: int = OBS_FLAT_SIZE,
                 action_size: int = ACTION_COUNT, timeout: float = 30.0):
        import gymnasium as gym

        self.env_num = num_bots
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), np.float32),
            "is_last": gym.spaces.Box(0, 1, (1,), np.float32),
            "is_terminal": gym.spaces.Box(0, 1, (1,), np.float32),
        })
        # One-hot action space: the trainer feeds the actor's one-hot (B, 41) output
        # straight through; we argmax it back to an integer index. `.discrete=True`
        # selects the onehot actor distribution in Dreamer.__init__.
        act = gym.spaces.Box(0, 1, (action_size,), np.float32)
        act.discrete = True
        self.action_space = act

        self._shm = ShmClient(num_bots, timeout=timeout)
        # is_first[i] is True on the first step after bot i was reset. C++ auto-resets
        # any done bot during the next step, so "reset" == "its previous output done".
        self._is_first = np.ones(num_bots, dtype=bool)

    def step(self, action, done):
        # action: (B, A) one-hot float tensor (any device); done: (B,) bool tensor.
        act = action.detach().cpu()
        done_np = done.detach().cpu().numpy().astype(bool) if torch.is_tensor(done) else np.asarray(done, dtype=bool)

        idx = act.argmax(dim=-1).numpy().astype(np.uint8)
        idx[done_np] = 0  # NOOP for envs the trainer considers done (C++ resets them)

        obs_flat, rewards, dones, comp = self._shm.step(idx)

        is_first = self._is_first.astype(np.float32)                 # fresh this step
        is_terminal = (comp[:, DEATH_COMPONENT] > 0).astype(np.float32)
        is_last = dones.astype(np.float32)
        self._is_first = dones                                        # fresh next step

        td = TensorDict({
            "obs": torch.as_tensor(obs_flat, dtype=torch.float32),       # (B, 1148)
            "is_first": torch.as_tensor(is_first),                       # (B,)
            "is_last": torch.as_tensor(is_last),                         # (B,)
            "is_terminal": torch.as_tensor(is_terminal),                 # (B,)
            "reward": torch.as_tensor(rewards, dtype=torch.float32),     # (B,)
        }, batch_size=(self.env_num,))

        # lift 1-D fields to (B, 1), matching ParallelEnv.lift_dim.
        td = TensorDict(
            {k: (v.unsqueeze(-1) if v.ndim == 1 else v) for k, v in td.items()},
            batch_size=(self.env_num,),
        )
        return td, torch.as_tensor(dones, dtype=torch.bool)

    def close(self):
        self._shm.close()
