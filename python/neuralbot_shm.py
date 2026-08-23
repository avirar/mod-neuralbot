"""
Shared-memory protocol client + observation projection, dependency-free (numpy only).

This is the single source of truth for the wire layout and the flat-obs
normalization. Imported by:
  - shared_memory_env.py  (SBX PPO path, async reader-thread pipeline)
  - wow_world_model_env.py (R2-Dreamer / NE-Dreamer spike, synchronous)

Protocol (shm v2): Python writes `num_bots` uint8 actions to the region, sets
`actions_ready=1`; the C++ world thread steps all bots and writes packed
`NeuralBotFrame` records, sets `obs_ready=1` and signals the eventfd.
"""
import mmap
import os
import struct
import time

import numpy as np

from neuralbot_client import (
    FRAME_DTYPE, FRAME_BYTES,
    NB_MAX_SPELLS, NB_MAX_QUESTS, NB_MAX_ENTITIES, NB_MAX_ITEMS,
)

SHM_NAME = "/neuralbot_shm"
SHM_PATH = "/dev/shm/neuralbot_shm"

# Must match NeuralBotSharedMem.h
SHM_MAGIC = 0x4E425348
SHM_MAX_BOTS = 4096
SHM_DONES_SIZE = SHM_MAX_BOTS
SHM_FRAME_REGION_SIZE = SHM_MAX_BOTS * FRAME_BYTES

SHM_OFFSET_CONTROL = 0
SHM_OFFSET_ACTIONS = 128
SHM_OFFSET_OBS = 0x2000

# NeuralBotSharedControl struct format (little-endian):
#   magic(4) version(4) num_bots(4) step_count(4)
#   eventfd(4) actions_ready(4) obs_ready(4) shutdown(4) frame_bytes(4)
#   _pad[28]
CTRL_FMT = "<IIIIiIIII28x"
CTRL_SIZE = struct.calcsize(CTRL_FMT)  # 64

# Flat policy observation = fixed-size projection of the structured frame.
SELF_FLAT = 26
TARGET_FLAT = 14
COUNTS_FLAT = 4
ENTITY_FLAT_PER = 8
SPELL_FLAT_PER = 7
QUEST_FLAT_PER = 6
ITEM_FLAT_PER = 3
OBS_FLAT_SIZE = (
    SELF_FLAT + TARGET_FLAT + COUNTS_FLAT
    + NB_MAX_ENTITIES * ENTITY_FLAT_PER
    + NB_MAX_SPELLS * SPELL_FLAT_PER
    + NB_MAX_QUESTS * QUEST_FLAT_PER
    + NB_MAX_ITEMS * ITEM_FLAT_PER
)

# ── Per-field normalization scales (v0.6.x) ─────────────────────────────────
# Raw magnitudes are wildly heterogeneous (health ~1e2, money ~1e6 copper, positions
# ~1e3, spell/entry IDs ~1e5, npcFlags up to 2^31); a vanilla MLP cannot fit such
# mixed scales. Ratios where a natural 0..1 exists; log1p for long-tailed magnitudes;
# fixed divisors for bounded quantities.
_LEVEL_CAP = 80.0
_POS_CAP = 10000.0
_ANGLE_CAP = 6.283185307179586  # 2π
_MAPID_CAP = 10000.0
_CLASS_CAP = 11.0
_COMBO_CAP = 5.0
_DIST_CAP = 60.0
_REACTION_CAP = 2.0
_TYPE_CAP = 3.0
_HEALTH_CAP = 100000.0
_MONEY_CAP = 1000000.0  # copper = 100 gold
_ID_CAP = 1000000.0
_NPCFLAG_CAP = float(2 ** 31)
_CD_CAP = 600000.0  # ms (10 min)
_COST_CAP = 10000.0
_RANGE_CAP = 100.0
_CAST_CAP = 10000.0
_STATUS_CAP = 5.0
_OBJ_CAP = 60.0
_QUALITY_CAP = 7.0


def _log1p_norm(x, cap):
    """log1p scaling: [0, cap] -> [0, 1] on a log scale, robust to 0."""
    return np.log1p(np.maximum(x, 0.0)) / np.log1p(cap)


def _frac(num, den):
    """num / max(den, 1) -> ~[0, 1] fraction."""
    return num / np.maximum(den, 1.0)


def flatten_frames(frames: np.ndarray) -> np.ndarray:
    """Project structured frames into a fixed-size float32 tensor (num_bots, OBS_FLAT_SIZE).

    Same field selection and OBS_FLAT_SIZE as before; a transformer / world model
    consuming the structured records directly can skip this projection.
    """
    num_bots = frames.shape[0]

    s = frames["self"]
    self_flat = np.stack([
        s["level"] / _LEVEL_CAP,
        _frac(s["health"], s["maxHealth"]),
        _log1p_norm(s["maxHealth"], _HEALTH_CAP),
        _frac(s["mana"], s["maxMana"]),
        _log1p_norm(s["maxMana"], _HEALTH_CAP),
        _frac(s["resource"], s["maxResource"]),
        _log1p_norm(s["maxResource"], _HEALTH_CAP),
        _frac(s["xp"], s["nextLevelXp"]),
        _log1p_norm(s["nextLevelXp"], _HEALTH_CAP),
        _log1p_norm(s["money"], _MONEY_CAP),
        s["posX"] / _POS_CAP, s["posY"] / _POS_CAP, s["posZ"] / _POS_CAP,
        s["orientation"] / _ANGLE_CAP,
        s["mapId"] / _MAPID_CAP, s["zoneId"] / _MAPID_CAP, s["areaId"] / _MAPID_CAP,
        s["alive"], s["inCombat"], s["moving"], s["casting"],
        s["inWater"], s["mounted"],
        s["classId"] / _CLASS_CAP, s["race"] / _CLASS_CAP,
        s["comboPoints"] / _COMBO_CAP,
    ], axis=-1)

    t = frames["target"]
    target_flat = np.stack([
        _log1p_norm(t["entry"], _ID_CAP),
        t["type"] / _TYPE_CAP,
        _frac(t["health"], t["maxHealth"]),
        _log1p_norm(t["maxHealth"], _HEALTH_CAP),
        t["level"] / _LEVEL_CAP,
        t["dx"] / _DIST_CAP, t["dy"] / _DIST_CAP, t["dz"] / _DIST_CAP,
        t["distance"] / _DIST_CAP,
        t["reaction"] / _REACTION_CAP,
        t["alive"], t["inCombat"], t["casting"],
        _log1p_norm(t["npcFlags"], _NPCFLAG_CAP),
    ], axis=-1)

    c = frames["counts"]
    counts_flat = np.stack([
        c["nSpells"] / NB_MAX_SPELLS, c["nQuests"] / NB_MAX_QUESTS,
        c["nEntities"] / NB_MAX_ENTITIES, c["nItems"] / NB_MAX_ITEMS,
    ], axis=-1)

    e = frames["entities"]
    entity_flat = np.stack([
        _log1p_norm(e["entry"], _ID_CAP),
        e["type"] / _TYPE_CAP,
        e["level"] / _LEVEL_CAP,
        _frac(e["health"], e["maxHealth"]),
        _log1p_norm(e["maxHealth"], _HEALTH_CAP),
        e["distance"] / _DIST_CAP,
        e["reaction"] / _REACTION_CAP,
        e["alive"],
    ], axis=-1).reshape(num_bots, -1)

    sp = frames["spells"]
    spell_flat = np.stack([
        _log1p_norm(sp["spellId"], _ID_CAP),
        _log1p_norm(sp["cooldownMs"], _CD_CAP),
        _log1p_norm(sp["cost"], _COST_CAP),
        sp["range"] / _RANGE_CAP, sp["minRange"] / _RANGE_CAP,
        _log1p_norm(sp["castTimeMs"], _CAST_CAP),
        sp["ready"],
    ], axis=-1).reshape(num_bots, -1)

    q = frames["quests"]
    quest_flat = np.stack([
        _log1p_norm(q["questId"], _ID_CAP),
        q["status"] / _STATUS_CAP,
        q["obj"][..., 0] / _OBJ_CAP, q["obj"][..., 1] / _OBJ_CAP,
        q["obj"][..., 2] / _OBJ_CAP, q["obj"][..., 3] / _OBJ_CAP,
    ], axis=-1).reshape(num_bots, -1)

    it = frames["items"]
    item_flat = np.stack([
        _log1p_norm(it["entry"], _ID_CAP),
        it["quality"] / _QUALITY_CAP,
        it["distance"] / _DIST_CAP,
    ], axis=-1).reshape(num_bots, -1)

    obs = np.concatenate([self_flat, target_flat, counts_flat,
                          entity_flat, spell_flat, quest_flat, item_flat], axis=-1)
    obs = obs.astype(np.float32)
    assert obs.shape == (num_bots, OBS_FLAT_SIZE)
    return obs


class ShmClient:
    """Synchronous shared-memory client: one batched step per call, no reader thread.

    The async pipeline in shared_memory_env.py is faster for PPO's fixed-rate rollout,
    but a world model needs strict obs↔action alignment for is_first/is_terminal
    bookkeeping, so this client steps synchronously.
    """

    def __init__(self, num_bots: int, timeout: float = 30.0):
        self.timeout = timeout

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

        ctrl = struct.unpack_from(CTRL_FMT, self._mm, SHM_OFFSET_CONTROL)
        magic, version, shm_num_bots, _step, _eventfd, _ar, _or_, _sd, frame_bytes = ctrl
        if magic != SHM_MAGIC:
            raise RuntimeError(f"Bad magic: 0x{magic:08X}")
        if frame_bytes != FRAME_BYTES:
            raise RuntimeError(f"Frame size mismatch: shm={frame_bytes} python={FRAME_BYTES}")

        self.num_bots = shm_num_bots
        if num_bots != self.num_bots:
            raise RuntimeError(f"num_bots={num_bots} != shm num_bots={self.num_bots}")

        self._actions = np.frombuffer(self._mm, dtype=np.uint8,
                                      count=self.num_bots, offset=SHM_OFFSET_ACTIONS)
        self._frames = np.frombuffer(self._mm, dtype=FRAME_DTYPE,
                                     count=self.num_bots, offset=SHM_OFFSET_OBS)
        self._dones = np.frombuffer(self._mm, dtype=np.uint8,
                                    count=self.num_bots,
                                    offset=SHM_OFFSET_OBS + SHM_FRAME_REGION_SIZE)

        self._ctrl_actions_ready_off = SHM_OFFSET_CONTROL + 20
        self._ctrl_obs_ready_off = SHM_OFFSET_CONTROL + 24

    def _get_flag(self, off: int) -> int:
        return struct.unpack_from("<I", self._mm, off)[0]

    def _set_flag(self, off: int, value: int):
        struct.pack_into("<I", self._mm, off, value)

    def step(self, actions: np.ndarray):
        """Write actions, block for the C++ batch step, return (obs, rewards, dones, components)."""
        self._actions[:] = actions.astype(np.uint8)
        self._set_flag(self._ctrl_actions_ready_off, 1)

        deadline = time.monotonic() + self.timeout
        while self._get_flag(self._ctrl_obs_ready_off) != 1:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for server step")
            time.sleep(0.0005)

        # Copy/flatten BEFORE releasing obs_ready, so C++ can't overwrite the mmap
        # region while we read.
        frames = self._frames[:self.num_bots]
        obs = flatten_frames(frames)
        rewards = np.array(frames["reward"]["total"], copy=True).astype(np.float32)
        components = np.array(frames["reward"]["components"], copy=True).astype(np.float32)
        dones = self._dones[:self.num_bots].copy().astype(bool)

        self._set_flag(self._ctrl_obs_ready_off, 0)
        return obs, rewards, dones, components

    def close(self):
        if getattr(self, "_mm", None):
            for attr in ("_actions", "_frames", "_dones"):
                if hasattr(self, attr):
                    delattr(self, attr)
            self._mm.close()
            self._mm = None
