import gymnasium as gym
from gymnasium import spaces
import numpy as np
from neuralbot_client import (
    NeuralBotClient,
    OBS_TOTAL_SIZE,
    ACTION_COUNT,
)


class WoWNeuralBotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, host="127.0.0.1", port=9000, bot_name="NeuralbotA", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.bot_name = bot_name
        self.client = NeuralBotClient(host, port, bot_name)

        self.observation_space = spaces.Box(
            low=-1.0, high=65535.0, shape=(OBS_TOTAL_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_COUNT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            obs = self.client.reset()
        except Exception:
            self.client.close()
            self.client.connect()
            obs = self.client.reset()
        info = {}
        return obs, info

    def step(self, action):
        try:
            obs, reward, terminated, info = self.client.step(int(action))
        except Exception:
            self.client.close()
            self.client.connect()
            obs = np.zeros(OBS_TOTAL_SIZE, dtype=np.float32)
            reward = 0.0
            terminated = True
            info = {}
        truncated = False
        if self.render_mode == "human":
            print(f"Action: {action} Reward: {reward:.4f} Done: {terminated}")
        return obs, reward, terminated, truncated, info

    def close(self):
        self.client.close()


gym.register(
    id="WoWNeuralBot-v0",
    entry_point="wow_neuralbot_env:WoWNeuralBotEnv",
)
