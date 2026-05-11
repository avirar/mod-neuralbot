"""
SharedMemoryVecEnv: replaces 400 threads + 400 TCP sockets with one mmap region.

Protocol:
  1. Python writes 400 actions to shared memory, sets actions_ready = 1
  2. C++ world thread processes all 400 bots, writes obs+reward+dones,
     clears actions_ready, sets obs_ready = 1, signals eventfd
  3. Python blocks on eventfd, reads all 400 obs (zero-copy numpy)

Single-threaded. No syscalls. No serialization.
"""
import mmap
import os
import struct
import numpy as np
from typing import Optional, Sequence

from neuralbot_client import OBS_TOTAL_SIZE, ACTION_COUNT, NUM_BOTS
from stable_baselines3.common.vec_env import VecEnv

SHM_NAME = "/neuralbot_shm"
SHM_PATH = "/dev/shm/neuralbot_shm"

# Must match NeuralBotSharedMem.h
SHM_MAGIC       = 0x4E425348
SHM_VERSION     = 1
SHM_MAX_BOTS    = 4096
SHM_OBS_PER_BOT = OBS_TOTAL_SIZE + 1 + 12  # obs[80] + reward + components[12] = 93
SHM_OBS_BYTES   = SHM_OBS_PER_BOT * 4
SHM_DONES_SIZE  = SHM_MAX_BOTS
SHM_OBS_REGION_SIZE = SHM_MAX_BOTS * SHM_OBS_BYTES

SHM_OFFSET_CONTROL = 0
SHM_OFFSET_ACTIONS = 128
SHM_OFFSET_OBS     = 0x2000

# NeuralBotSharedControl struct format (little-endian):
#   magic(4) version(4) num_bots(4) step_count(4)
#   eventfd(4) actions_ready(4) obs_ready(4) shutdown(4)
#   _pad[28]
CTRL_FMT = "<IIIIiIII32x"
CTRL_SIZE = struct.calcsize(CTRL_FMT)  # 64


class SharedMemoryVecEnv(VecEnv):
    """VecEnv-compatible wrapper using shared memory for batch stepping."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._closed = False

        self._connect()
        self._setup_regions()

        self.num_envs = self.num_bots

        import gymnasium as gym
        observation_space = gym.spaces.Box(
            low=-1.0, high=65535.0, shape=(OBS_TOTAL_SIZE,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(ACTION_COUNT)
        super().__init__(self.num_envs, observation_space, action_space)

    def _connect(self):
        # Wait for shared memory to appear
        import time
        for _ in range(100):
            if os.path.exists(SHM_PATH):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Shared memory '{SHM_PATH}' not found after 10s")

        fd = os.open(SHM_PATH, os.O_RDWR)
        self._shm_size = os.fstat(fd).st_size
        self._mm = mmap.mmap(fd, self._shm_size, access=mmap.ACCESS_WRITE)
        os.close(fd)

        # Parse control struct
        ctrl = struct.unpack_from(CTRL_FMT, self._mm, SHM_OFFSET_CONTROL)
        magic, version, num_bots, step_count, eventfd, actions_ready, obs_ready, shutdown = ctrl

        if magic != SHM_MAGIC:
            raise RuntimeError(f"Bad magic: 0x{magic:08X}, expected 0x{SHM_MAGIC:08X}")
        if version != SHM_VERSION:
            raise RuntimeError(f"Version mismatch: {version} vs {SHM_VERSION}")

        self.num_bots  = num_bots
        self.eventfd   = eventfd

        assert self.num_bots <= SHM_MAX_BOTS
        assert self.num_bots == NUM_BOTS, f"num_bots={self.num_bots} != NUM_BOTS={NUM_BOTS}"

    def _setup_regions(self):
        # Actions: write as uint8 array
        self._actions = np.frombuffer(
            self._mm, dtype=np.uint8,
            count=self.num_bots, offset=SHM_OFFSET_ACTIONS
        )

        # Observations: read as float32 zero-copy view
        obs_buf = np.frombuffer(
            self._mm, dtype=np.float32,
            count=self.num_bots * SHM_OBS_PER_BOT, offset=SHM_OFFSET_OBS
        )
        self._obs_all = obs_buf.reshape(self.num_bots, SHM_OBS_PER_BOT)

        # Dones: uint8
        self._dones = np.frombuffer(
            self._mm, dtype=np.uint8,
            count=self.num_bots,
            offset=SHM_OFFSET_OBS + SHM_OBS_REGION_SIZE
        )

        # Control field offsets
        self._ctrl_actions_ready_off = SHM_OFFSET_CONTROL + 20
        self._ctrl_obs_ready_off     = SHM_OFFSET_CONTROL + 24
        self._ctrl_shutdown_off      = SHM_OFFSET_CONTROL + 28
        self._ctrl_step_count_off    = SHM_OFFSET_CONTROL + 12

    # ─── VecEnv API ──────────────────────────────────────────────────

    def step_async(self, actions: np.ndarray):
        """Write all actions to shared memory and signal the server."""
        self._actions[:] = actions.astype(np.uint8)
        # Write actions_ready = 1 (Python→C++ direction)
        struct.pack_into("<I", self._mm, self._ctrl_actions_ready_off, 1)

    def step_wait(self):
        """Block until server processes actions, then read results."""
        self._wait_eventfd()

        # Read results (copy to avoid aliasing — C++ may overwrite next round)
        obs = np.array(self._obs_all[:, :OBS_TOTAL_SIZE], copy=True)
        rewards = np.array(self._obs_all[:, OBS_TOTAL_SIZE], copy=True)
        components = np.array(self._obs_all[:, OBS_TOTAL_SIZE + 1:], copy=True)
        dones = self._dones[:self.num_bots].copy().astype(bool)

        # Clear obs_ready so C++ can signal next time
        self._set_flag("obs_ready", 0)

        infos = self._build_infos(components, dones)
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray):
        """Combined step_async + step_wait."""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        """No individual reset needed. Returns dummy obs for API compat."""
        return np.zeros((self.num_envs, OBS_TOTAL_SIZE), dtype=np.float32)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._set_flag("shutdown", 1)
        if hasattr(self, '_mm') and self._mm:
            # Release numpy views before closing mmap to avoid BufferError
            for attr in ('_actions', '_obs_all', '_dones'):
                if hasattr(self, attr):
                    delattr(self, attr)
            self._mm.close()
            self._mm = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    # ─── Helpers ─────────────────────────────────────────────────────

    def _build_infos(self, components: np.ndarray, dones: np.ndarray) -> list:
        keys = [
            "xp", "damage_taken", "kill", "death", "loot",
            "quest_accepted", "quest_completed", "quest_proximity",
            "quest_progress", "enemy_proximity", "target_acquired", "time_penalty",
        ]
        infos = []
        for i in range(self.num_bots):
            info = {}
            if not dones[i]:
                info["reward_components"] = {k: float(components[i, j]) for j, k in enumerate(keys)}
            infos.append(info)
        return infos

    def _set_flag(self, name: str, value: int):
        off = getattr(self, f"_ctrl_{name}_off", None)
        if off is not None:
            struct.pack_into("<I", self._mm, off, value)

    def _get_flag(self, name: str) -> int:
        off = getattr(self, f"_ctrl_{name}_off", None)
        if off is None:
            return 0
        return struct.unpack_from("<I", self._mm, off)[0]

    def _wait_eventfd(self):
        """Block until server signals completion (via eventfd or polling)."""
        # Try eventfd first; fd may be invalid cross-process, fall back to polling
        if self.eventfd >= 0:
            try:
                os.eventfd_read(self.eventfd)
                return
            except (OSError, BlockingIOError):
                pass  # fall through to polling

        # Poll the obs_ready flag (server writes it, we read it)
        import time
        start = time.monotonic()
        while self._get_flag("obs_ready") != 1:
            if time.monotonic() - start > self.timeout:
                raise TimeoutError("Timed out waiting for server step")
            time.sleep(0.001)

    def _get_step_count(self) -> int:
        return struct.unpack_from("<I", self._mm, self._ctrl_step_count_off)[0]
