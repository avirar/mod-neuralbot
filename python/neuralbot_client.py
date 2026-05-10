import socket
import struct
import numpy as np

OBS_PLAYER_STATE_SIZE = 20
OBS_NEARBY_UNITS_COUNT = 10
OBS_NEARBY_UNIT_FEATURES = 8
OBS_COMBAT_STATE_SIZE = 10
OBS_QUEST_STATE_SIZE = 10
OBS_OPCODE_HISTORY_SIZE = 64
OBS_TOTAL_SIZE = (
    OBS_PLAYER_STATE_SIZE
    + OBS_NEARBY_UNITS_COUNT * OBS_NEARBY_UNIT_FEATURES
    + OBS_COMBAT_STATE_SIZE
    + OBS_QUEST_STATE_SIZE
    + OBS_OPCODE_HISTORY_SIZE
)

ACTION_NOOP = 0
ACTION_MOVE_FORWARD = 1
ACTION_MOVE_BACKWARD = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_RIGHT = 4
ACTION_STOP_MOVE = 9
ACTION_TARGET_NEAREST_ENEMY = 12
ACTION_ATTACK_START = 17
ACTION_ATTACK_STOP = 18
ACTION_CAST_SPELL_1 = 19
ACTION_CAST_SPELL_2 = 20
ACTION_CAST_SPELL_3 = 21
ACTION_INTERACT_LOOT = 22
ACTION_STAND_UP = 23
ACTION_INTERACT_NPC = 24
ACTION_COMPLETE_QUEST = 25
ACTION_TARGET_QUEST_GIVER = 26

ACTION_COUNT = 27

ACTION_NAMES = {
    0: "NOOP",
    1: "MOVE_FORWARD",
    2: "MOVE_BACKWARD",
    3: "MOVE_LEFT",
    4: "MOVE_RIGHT",
    5: "MOVE_FWD_LEFT",
    6: "MOVE_FWD_RIGHT",
    7: "MOVE_BWD_LEFT",
    8: "MOVE_BWD_RIGHT",
    9: "STOP_MOVE",
    10: "TURN_LEFT",
    11: "TURN_RIGHT",
    12: "TARGET_NEAREST",
    13: "TARGET_0",
    14: "TARGET_1",
    15: "TARGET_2",
    16: "TARGET_3",
    17: "ATTACK_START",
    18: "ATTACK_STOP",
    19: "CAST_1",
    20: "CAST_2",
    21: "CAST_3",
    22: "LOOT",
    23: "STAND_UP",
    24: "INTERACT_NPC",
    25: "COMPLETE_QUEST",
    26: "TARGET_QUEST_GIVER",
}


BOT_NAMES = [f"Neuralbot{i}" for i in range(20)]


class NeuralBotClient:
    def __init__(self, host="127.0.0.1", port=9000, bot_name="Neuralbot0"):
        self.host = host
        self.port = port
        self.bot_name = bot_name
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._sock.settimeout(30.0)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, msg: str) -> str:
        self._sock.sendall((msg + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            data += chunk
        return data.decode("utf-8").strip()

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
        if len(extra) >= 10:
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
                "time_penalty": extra[9],
            }
        return obs, reward, done, info
