import os
import socket
import struct
import numpy as np

OBS_PLAYER_STATE_SIZE = 20
OBS_NEARBY_UNITS_COUNT = 5
OBS_NEARBY_UNIT_FEATURES = 8
OBS_COMBAT_STATE_SIZE = 15
OBS_QUEST_STATE_SIZE = 10
OBS_TOTAL_SIZE = (
    OBS_PLAYER_STATE_SIZE
    + OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES
    + OBS_COMBAT_STATE_SIZE
    + OBS_QUEST_STATE_SIZE
)

# Action space v2 (must mirror NeuralBotCommon.h)
ACTION_NOOP = 0
ACTION_MOVE_TO_TARGET = 1
ACTION_STOP_MOVE = 2
ACTION_MOVE_FORWARD = 3
ACTION_MOVE_BACKWARD = 4
ACTION_TURN_LEFT = 5
ACTION_TURN_RIGHT = 6
ACTION_TARGET_NEAREST_ENEMY = 7
ACTION_TARGET_NEAREST_FRIENDLY = 8
ACTION_TARGET_NEAREST_CORPSE = 9
ACTION_TARGET_ENTITY_0 = 10
MAX_ENTITY_TARGET_SLOTS = 18
ACTION_ATTACK_START = 28
ACTION_ATTACK_STOP = 29
ACTION_CAST_SPELL_0 = 30
MAX_CAST_SLOTS = 8
ACTION_INTERACT_TARGET = 38
ACTION_COMPLETE_QUEST = 39
ACTION_LOOT = 40

ACTION_COUNT = 41


def _build_action_names():
    names = {
        0: "NOOP", 1: "MOVE_TO_TARGET", 2: "STOP_MOVE", 3: "MOVE_FORWARD",
        4: "MOVE_BACKWARD", 5: "TURN_LEFT", 6: "TURN_RIGHT",
        7: "TARGET_NEAREST_ENEMY", 8: "TARGET_NEAREST_FRIENDLY", 9: "TARGET_NEAREST_CORPSE",
        28: "ATTACK_START", 29: "ATTACK_STOP",
        38: "INTERACT_TARGET", 39: "COMPLETE_QUEST", 40: "LOOT",
    }
    for i in range(MAX_ENTITY_TARGET_SLOTS):
        names[ACTION_TARGET_ENTITY_0 + i] = f"TARGET_ENTITY_{i}"
    for i in range(MAX_CAST_SLOTS):
        names[ACTION_CAST_SPELL_0 + i] = f"CAST_SPELL_{i}"
    return names


ACTION_NAMES = _build_action_names()


NUM_BOTS = int(os.environ.get("NEURALBOT_NUM_BOTS", "400"))

def generate_bot_name(index: int) -> str:
    if index < 26:
        return f"Neuralbot{chr(ord('A') + index)}"
    i = index - 26
    return f"Neuralbot{chr(ord('A') + i // 26)}{chr(ord('A') + i % 26)}"

BOT_NAMES = [generate_bot_name(i) for i in range(NUM_BOTS)]


# ── Faithful structured frame (shm protocol v2) ─────────────────────────
# Mirrors src/NeuralBotFrame.h byte-for-byte. All dtypes use the numpy default
# align=False (packed), matching the C++ #pragma pack(1) wire structs. FRAME_BYTES
# is cross-checked against the control block's `frame_bytes` field at connect time.

SHM_VERSION = 2

NB_MAX_SPELLS = 64
NB_MAX_QUESTS = 16
NB_MAX_ENTITIES = 64
NB_MAX_ITEMS = 16
NB_REWARD_COMPONENTS = 14

NB_ENTITY_TYPE_NONE = 0
NB_ENTITY_TYPE_CREATURE = 1
NB_ENTITY_TYPE_PLAYER = 2
NB_ENTITY_TYPE_GAMEOBJECT = 3

NB_REACTION_NEUTRAL = 0
NB_REACTION_HOSTILE = 1
NB_REACTION_FRIENDLY = 2

SELF_DTYPE = np.dtype([
    ("guid", "<u8"),
    ("level", "<u4"),
    ("health", "<f4"), ("maxHealth", "<f4"),
    ("mana", "<f4"), ("maxMana", "<f4"),
    ("resource", "<f4"), ("maxResource", "<f4"),
    ("xp", "<u4"), ("nextLevelXp", "<u4"),
    ("money", "<u4"),
    ("posX", "<f4"), ("posY", "<f4"), ("posZ", "<f4"), ("orientation", "<f4"),
    ("mapId", "<u4"), ("zoneId", "<u4"), ("areaId", "<u4"),
    ("alive", "<u1"), ("inCombat", "<u1"), ("moving", "<u1"), ("casting", "<u1"),
    ("inWater", "<u1"), ("mounted", "<u1"), ("classId", "<u1"), ("race", "<u1"),
    ("comboPoints", "<u4"),
    ("targetGuid", "<u8"),
])

TARGET_DTYPE = np.dtype([
    ("guid", "<u8"),
    ("entry", "<u4"),
    ("type", "<u1"),
    ("health", "<f4"), ("maxHealth", "<f4"),
    ("level", "<u4"),
    ("dx", "<f4"), ("dy", "<f4"), ("dz", "<f4"), ("distance", "<f4"),
    ("reaction", "<u1"), ("alive", "<u1"), ("inCombat", "<u1"), ("casting", "<u1"),
    ("npcFlags", "<u4"),
])

COUNTS_DTYPE = np.dtype([
    ("nSpells", "<u2"), ("nQuests", "<u2"), ("nEntities", "<u2"), ("nItems", "<u2"),
])

SPELL_DTYPE = np.dtype([
    ("spellId", "<u4"), ("cooldownMs", "<u4"), ("cost", "<u4"),
    ("range", "<f4"), ("minRange", "<f4"), ("castTimeMs", "<f4"),
    ("ready", "<u1"), ("pad", "<u1", (3,)),
])

QUEST_DTYPE = np.dtype([
    ("questId", "<u4"), ("status", "<u1"), ("pad", "<u1", (3,)),
    ("obj", "<u2", (4,)),
])

ENTITY_DTYPE = np.dtype([
    ("guid", "<u8"), ("entry", "<u4"), ("type", "<u1"), ("pad", "<u1", (3,)),
    ("level", "<u4"),
    ("health", "<f4"), ("maxHealth", "<f4"),
    ("dx", "<f4"), ("dy", "<f4"), ("dz", "<f4"), ("distance", "<f4"),
    ("reaction", "<u1"), ("alive", "<u1"), ("inCombat", "<u1"), ("casting", "<u1"),
    ("npcFlags", "<u4"),
])

ITEM_DTYPE = np.dtype([
    ("guid", "<u8"), ("entry", "<u4"), ("quality", "<u1"), ("pad", "<u1", (3,)),
    ("distance", "<f4"),
])

REWARD_DTYPE = np.dtype([
    ("total", "<f4"),
    ("components", "<f4", (NB_REWARD_COMPONENTS,)),
])

FRAME_DTYPE = np.dtype([
    ("self", SELF_DTYPE),
    ("target", TARGET_DTYPE),
    ("counts", COUNTS_DTYPE),
    ("spells", SPELL_DTYPE, (NB_MAX_SPELLS,)),
    ("quests", QUEST_DTYPE, (NB_MAX_QUESTS,)),
    ("entities", ENTITY_DTYPE, (NB_MAX_ENTITIES,)),
    ("items", ITEM_DTYPE, (NB_MAX_ITEMS,)),
    ("reward", REWARD_DTYPE),
])

FRAME_BYTES = FRAME_DTYPE.itemsize

REWARD_COMPONENT_KEYS = [
    "xp", "damage_taken", "kill", "death", "loot",
    "quest_accepted", "quest_completed", "quest_proximity",
    "quest_progress", "enemy_proximity", "target_acquired",
    "spell_learned", "trainer_proximity", "time_penalty",
]


class NeuralBotClient:
    def __init__(self, host="127.0.0.1", port=9000, bot_name="NeuralbotA"):
        self.host = host
        self.port = port
        self.bot_name = bot_name
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._sock.settimeout(60.0)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, msg: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                self._sock.sendall((msg + "\n").encode("utf-8"))
                data = b""
                while b"\n" not in data:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("Server closed connection")
                    data += chunk
                return data.decode("utf-8").strip()
            except (ConnectionError, OSError):
                if attempt < retries - 1:
                    self.close()
                    self.connect()
                else:
                    raise

    def ping(self) -> bool:
        resp = self._send("PING")
        return resp == "PONG"

    def status(self) -> dict:
        resp = self._send(f"STATUS {self.bot_name}")
        parts = resp.split()
        result = {"state": parts[1] if len(parts) > 1 else "UNKNOWN"}
        if len(parts) > 4:
            result["name"] = parts[2]
            result["level"] = int(parts[3])
            result["zone"] = int(parts[4])
        return result

    def reset(self) -> np.ndarray:
        resp = self._send(f"RESET {self.bot_name}")
        return self._parse_obs(resp)

    def step(self, action: int):
        resp = self._send(f"STEP {self.bot_name} {action}")
        return self._parse_step_result(resp)

    def get_spellbook(self) -> list:
        resp = self._send(f"SPELLS {self.bot_name}")
        parts = resp.split()
        if parts[0] == "SPELLS" and len(parts) > 1:
            return [int(x) for x in parts[1:]]
        return []

    def send_spellbook(self, spell_ids: list):
        spell_str = " ".join(str(s) for s in spell_ids)
        self._send(f"SEND_SPELLBOOK {self.bot_name} {spell_str}")
        return "OK"

    def set_spells(self, spell_ids: list):
        spell_str = " ".join(str(s) for s in spell_ids)
        self._send(f"SET_SPELLS {self.bot_name} {spell_str}")

    def _parse_obs(self, resp: str) -> np.ndarray:
        parts = resp.split()
        if parts[0] == "OBS":
            floats = [float(x) for x in parts[1:]]
            return np.array(floats[:OBS_TOTAL_SIZE], dtype=np.float32)
        return np.zeros(OBS_TOTAL_SIZE, dtype=np.float32)

    def _parse_step_result(self, resp: str):
        parts = resp.split()
        done = parts[1] == "1" if len(parts) > 1 else False
        reward = float(parts[2]) if len(parts) > 2 else 0.0
        floats = [float(x) for x in parts[3:]]
        obs = np.array(floats[:OBS_TOTAL_SIZE], dtype=np.float32)

        info = {}
        extra = floats[OBS_TOTAL_SIZE:]
        if len(extra) >= 12:
            info["reward_components"] = {
                "xp": extra[0],
                "damage_taken": extra[1],
                "kill": extra[2],
                "death": extra[3],
                "loot": extra[4],
                "quest_accepted": extra[5],
                "quest_completed": extra[6],
                "quest_proximity": extra[7],
                "quest_progress": extra[8],
                "enemy_proximity": extra[9],
                "target_acquired": extra[10],
                "time_penalty": extra[11],
            }
        return obs, reward, done, info
