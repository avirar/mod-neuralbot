"""
SharedMemoryVecEnv: replaces 400 threads + 400 TCP sockets with one mmap region.

Protocol (shm v2 — faithful structured frames):
  1. Python writes 400 actions to shared memory, sets actions_ready = 1
  2. C++ world thread processes all 400 bots, writes packed NeuralBotFrame records,
     clears actions_ready, sets obs_ready = 1, signals eventfd
  3. Python blocks on eventfd, reads all 400 frames (zero-copy numpy structured views)

Single-threaded. No syscalls. No serialization.

The wire format is the structured, entity-centric `NeuralBotFrame` (see
src/NeuralBotFrame.h / neuralbot_client.py). The policy-facing observation is a fixed-size
flattened projection (`OBS_FLAT_SIZE`) built from the structured records — a transformer /
DreamerV3 policy (ROADMAP §5) will consume the records directly instead.
"""
import mmap
import os
import struct
import threading
import time
import numpy as np
from collections import deque
from typing import Optional, Sequence

from neuralbot_client import (
    ACTION_COUNT, NUM_BOTS, SHM_VERSION, FRAME_DTYPE, FRAME_BYTES,
    NB_MAX_SPELLS, NB_MAX_QUESTS, NB_MAX_ENTITIES, NB_MAX_ITEMS,
    NB_REWARD_COMPONENTS, REWARD_COMPONENT_KEYS,
)
from stable_baselines3.common.vec_env import VecEnv

SHM_NAME = "/neuralbot_shm"
SHM_PATH = "/dev/shm/neuralbot_shm"

# Must match NeuralBotSharedMem.h
SHM_MAGIC       = 0x4E425348
SHM_MAX_BOTS    = 4096
SHM_DONES_SIZE  = SHM_MAX_BOTS
SHM_FRAME_REGION_SIZE = SHM_MAX_BOTS * FRAME_BYTES

SHM_OFFSET_CONTROL = 0
SHM_OFFSET_ACTIONS = 128
SHM_OFFSET_OBS     = 0x2000

# NeuralBotSharedControl struct format (little-endian):
#   magic(4) version(4) num_bots(4) step_count(4)
#   eventfd(4) actions_ready(4) obs_ready(4) shutdown(4) frame_bytes(4)
#   _pad[28]
CTRL_FMT = "<IIIIiIIII28x"
CTRL_SIZE = struct.calcsize(CTRL_FMT)  # 64

# Flat policy observation = fixed-size projection of the structured frame.
SELF_FLAT       = 26
TARGET_FLAT     = 14
COUNTS_FLAT     = 4
ENTITY_FLAT_PER = 8
SPELL_FLAT_PER  = 7
QUEST_FLAT_PER  = 6
ITEM_FLAT_PER   = 3
OBS_FLAT_SIZE = (
    SELF_FLAT + TARGET_FLAT + COUNTS_FLAT
    + NB_MAX_ENTITIES * ENTITY_FLAT_PER
    + NB_MAX_SPELLS * SPELL_FLAT_PER
    + NB_MAX_QUESTS * QUEST_FLAT_PER
    + NB_MAX_ITEMS * ITEM_FLAT_PER
)


def flatten_frames(frames: np.ndarray) -> np.ndarray:
    """Project structured frames into a fixed-size float32 tensor (num_bots, OBS_FLAT_SIZE)."""
    num_bots = frames.shape[0]

    s = frames["self"]
    self_flat = np.stack([
        s["level"].astype(np.float32),
        s["health"], s["maxHealth"],
        s["mana"], s["maxMana"],
        s["resource"], s["maxResource"],
        s["xp"].astype(np.float32), s["nextLevelXp"].astype(np.float32),
        s["money"].astype(np.float32),
        s["posX"], s["posY"], s["posZ"], s["orientation"],
        s["mapId"].astype(np.float32), s["zoneId"].astype(np.float32), s["areaId"].astype(np.float32),
        s["alive"].astype(np.float32), s["inCombat"].astype(np.float32),
        s["moving"].astype(np.float32), s["casting"].astype(np.float32),
        s["inWater"].astype(np.float32), s["mounted"].astype(np.float32),
        s["classId"].astype(np.float32), s["race"].astype(np.float32),
        s["comboPoints"].astype(np.float32),
    ], axis=-1)

    t = frames["target"]
    target_flat = np.stack([
        t["entry"].astype(np.float32), t["type"].astype(np.float32),
        t["health"], t["maxHealth"], t["level"].astype(np.float32),
        t["dx"], t["dy"], t["dz"], t["distance"],
        t["reaction"].astype(np.float32), t["alive"].astype(np.float32),
        t["inCombat"].astype(np.float32), t["casting"].astype(np.float32),
        t["npcFlags"].astype(np.float32),
    ], axis=-1)

    c = frames["counts"]
    counts_flat = np.stack([
        c["nSpells"].astype(np.float32), c["nQuests"].astype(np.float32),
        c["nEntities"].astype(np.float32), c["nItems"].astype(np.float32),
    ], axis=-1)

    e = frames["entities"]
    entity_flat = np.stack([
        e["entry"].astype(np.float32), e["type"].astype(np.float32),
        e["level"].astype(np.float32), e["health"], e["maxHealth"],
        e["distance"], e["reaction"].astype(np.float32), e["alive"].astype(np.float32),
    ], axis=-1).reshape(num_bots, -1)

    sp = frames["spells"]
    spell_flat = np.stack([
        sp["spellId"].astype(np.float32), sp["cooldownMs"].astype(np.float32),
        sp["cost"].astype(np.float32), sp["range"], sp["minRange"],
        sp["castTimeMs"], sp["ready"].astype(np.float32),
    ], axis=-1).reshape(num_bots, -1)

    q = frames["quests"]
    quest_flat = np.stack([
        q["questId"].astype(np.float32), q["status"].astype(np.float32),
        q["obj"][..., 0].astype(np.float32), q["obj"][..., 1].astype(np.float32),
        q["obj"][..., 2].astype(np.float32), q["obj"][..., 3].astype(np.float32),
    ], axis=-1).reshape(num_bots, -1)

    it = frames["items"]
    item_flat = np.stack([
        it["entry"].astype(np.float32), it["quality"].astype(np.float32), it["distance"],
    ], axis=-1).reshape(num_bots, -1)

    obs = np.concatenate([self_flat, target_flat, counts_flat,
                          entity_flat, spell_flat, quest_flat, item_flat], axis=-1)
    assert obs.shape == (num_bots, OBS_FLAT_SIZE)
    return obs


# Pipelined harvest queue: the reader thread continuously pulls frame batches into a
# small deque so the trainer consumes 1-tick-stale observations (standard frame-skip
# semantics) while C++ never idles waiting for Python. C++ backpressures on obs_ready.
QUEUE_MAXLEN = 3


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
            low=-1.0, high=65535.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(ACTION_COUNT)
        super().__init__(self.num_envs, observation_space, action_space)

        # ── reader thread: harvest frames as fast as C++ produces them ──
        self._queue: deque = deque(maxlen=QUEUE_MAXLEN)
        self._cond = threading.Condition()
        self._reader_stop = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

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
        magic, version, num_bots, step_count, eventfd, actions_ready, obs_ready, shutdown, frame_bytes = ctrl

        if magic != SHM_MAGIC:
            raise RuntimeError(f"Bad magic: 0x{magic:08X}, expected 0x{SHM_MAGIC:08X}")
        if version != SHM_VERSION:
            raise RuntimeError(f"Version mismatch: {version} vs {SHM_VERSION}")
        if frame_bytes != FRAME_BYTES:
            raise RuntimeError(f"Frame size mismatch: shm={frame_bytes} python={FRAME_BYTES}")

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

        # Observations: structured NeuralBotFrame records, zero-copy view
        self._frames = np.frombuffer(
            self._mm, dtype=FRAME_DTYPE,
            count=self.num_bots, offset=SHM_OFFSET_OBS
        )

        # Dones: uint8
        self._dones = np.frombuffer(
            self._mm, dtype=np.uint8,
            count=self.num_bots,
            offset=SHM_OFFSET_OBS + SHM_FRAME_REGION_SIZE
        )

        # Control field offsets
        self._ctrl_actions_ready_off = SHM_OFFSET_CONTROL + 20
        self._ctrl_obs_ready_off     = SHM_OFFSET_CONTROL + 24
        self._ctrl_shutdown_off      = SHM_OFFSET_CONTROL + 28
        self._ctrl_step_count_off    = SHM_OFFSET_CONTROL + 12

    # ─── VecEnv API ──────────────────────────────────────────────────

    def _reader_loop(self):
        """Continuously harvest frame batches. Copies everything out, clears obs_ready
        immediately (releases C++ backpressure), pushes into the queue."""
        while not self._reader_stop.is_set():
            if self._get_flag("obs_ready") != 1:
                time.sleep(0.0002)
                continue
            try:
                frames = self._frames[:self.num_bots]
                obs = flatten_frames(frames)
                rewards = np.array(frames["reward"]["total"], copy=True)
                components = np.array(frames["reward"]["components"], copy=True)
                dones = self._dones[:self.num_bots].copy().astype(bool)
            except BufferError:
                continue
            # Release C++ *after* copying — the backpressure guard in ProcessSharedMemoryStep
            self._set_flag("obs_ready", 0)
            with self._cond:
                if len(self._queue) == self._queue.maxlen:
                    self._queue.popleft()  # drop oldest rather than block the harvester
                self._queue.append((obs, rewards, dones, components))
                self._cond.notify()

    def _pop_result(self):
        deadline = time.monotonic() + self.timeout
        with self._cond:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for server step")
                self._cond.wait(timeout=min(remaining, 1.0))
            return self._queue.popleft()

    def step_async(self, actions: np.ndarray):
        """Write all actions to shared memory and signal the server."""
        self._actions[:] = actions.astype(np.uint8)
        # Write actions_ready = 1 (Python→C++ direction)
        struct.pack_into("<I", self._mm, self._ctrl_actions_ready_off, 1)

    def step_wait(self):
        """Return the next harvested batch (obs may be 1 tick stale — pipelined)."""
        obs, rewards, dones, components = self._pop_result()
        infos = self._build_infos(components)
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray):
        """Combined step_async + step_wait."""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        """No individual reset needed. Returns dummy obs for API compat."""
        return np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, '_reader_stop'):
            self._reader_stop.set()
            if hasattr(self, '_reader') and self._reader.is_alive():
                self._reader.join(timeout=2.0)
        self._set_flag("shutdown", 1)
        if hasattr(self, '_mm') and self._mm:
            # Release numpy views before closing mmap to avoid BufferError
            for attr in ('_actions', '_frames', '_dones'):
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

    def _build_infos(self, components: np.ndarray) -> list:
        infos = []
        for i in range(self.num_bots):
            info = {}
            info["reward_components"] = {
                key: float(components[i, j]) for j, key in enumerate(REWARD_COMPONENT_KEYS)
            }
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
