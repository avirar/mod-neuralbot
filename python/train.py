#!/usr/bin/env python3
import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from wow_neuralbot_env import WoWNeuralBotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from neuralbot_client import ACTION_COUNT, BOT_NAMES


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, stats_path: str, verbose=0):
        super().__init__(verbose)
        self.stats_path = stats_path
        self.episode_num = 0
        self.rows = []
        self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int32)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions", [])
        infos = self.locals.get("infos", [])

        for i in range(len(dones)):
            if i < len(actions):
                self.current_actions[int(actions[i])] += 1

            if dones[i] and i < len(infos):
                ep_info = infos[i].get("episode", {})
                rc = infos[i].get("reward_components", {})

                row = {
                    "episode": self.episode_num,
                    "reward": round(ep_info.get("r", 0.0), 4),
                    "length": ep_info.get("l", 0),
                    "xp": round(rc.get("xp", 0.0), 4),
                    "kill": round(rc.get("kill", 0.0), 4),
                    "death": round(rc.get("death", 0.0), 4),
                    "quest_accepted": round(rc.get("quest_accepted", 0.0), 4),
                    "quest_completed": round(rc.get("quest_completed", 0.0), 4),
                    "quest_proximity": round(rc.get("quest_proximity", 0.0), 4),
                    "quest_progress": round(rc.get("quest_progress", 0.0), 4),
                }
                for a in range(ACTION_COUNT):
                    row[f"act_{a}"] = int(self.current_actions[a])

                self.rows.append(row)
                self.episode_num += 1
                self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int32)
                self._write_csv()

        return True

    def _write_csv(self):
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        with open(self.stats_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def make_env(bot_name, host, port):
    def _init():
        return WoWNeuralBotEnv(host=host, port=port, bot_name=bot_name)
    return _init


def main():
    host = os.environ.get("NEURALBOT_HOST", "127.0.0.1")
    port = int(os.environ.get("NEURALBOT_PORT", "9000"))
    timesteps = int(os.environ.get("NEURALBOT_TIMESTEPS", "5000000"))
    model_path = os.environ.get("NEURALBOT_MODEL", "wow_neuralbot_model")
    stats_path = model_path + "_episodes.csv"

    print(f"Creating SubprocVecEnv with {len(BOT_NAMES)} bots...")
    env = SubprocVecEnv([make_env(name, host, port) for name in BOT_NAMES])

    obs = env.reset()

    print(f"Training PPO for {timesteps} timesteps using {len(BOT_NAMES)} parallel envs...")
    print(f"Episode stats: {stats_path}")

    checkpoint_path = os.path.dirname(model_path) or "."
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_path,
        name_prefix="wow_neuralbot",
    )
    stats_callback = EpisodeStatsCallback(stats_path)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, stats_callback],
        progress_bar=True,
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()


if __name__ == "__main__":
    main()
