#!/usr/bin/env python3
"""
Training script v2: SBX PPO (JAX) + ThreadedVecEnv for maximum throughput.

Scales to 400+ bots with parallel TCP via threads.
SBX uses JAX JIT-compiled gradient updates for 10-20x faster training.
"""
import sys
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from wow_neuralbot_env import WoWNeuralBotEnv
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from threaded_vec_env import ThreadedVecEnv
from neuralbot_client import ACTION_COUNT, BOT_NAMES, NUM_BOTS


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, stats_path: str, verbose=0):
        super().__init__(verbose)
        self.stats_path = stats_path
        self.episode_num = 0
        self.rows = []
        self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int32)
        self._ep_rewards = [0.0] * NUM_BOTS
        self._ep_lengths = [0] * NUM_BOTS
        self._ep_xp = [0.0] * NUM_BOTS
        self._ep_kill = [0.0] * NUM_BOTS
        self._ep_enemy_prox = [0.0] * NUM_BOTS
        self._ep_target_acq = [0.0] * NUM_BOTS
        self._ep_quest_prox = [0.0] * NUM_BOTS

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions", [])
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])

        for i in range(len(dones)):
            if i < len(actions):
                self.current_actions[int(actions[i])] += 1
            if i < len(rewards):
                self._ep_rewards[i] += float(rewards[i])
                self._ep_lengths[i] += 1
            if i < len(infos):
                rc = infos[i].get("reward_components", {})
                self._ep_xp[i] += rc.get("xp", 0.0)
                self._ep_kill[i] += rc.get("kill", 0.0)
                self._ep_enemy_prox[i] += rc.get("enemy_proximity", 0.0)
                self._ep_target_acq[i] += rc.get("target_acquired", 0.0)
                self._ep_quest_prox[i] += rc.get("quest_proximity", 0.0)

            if dones[i] and i < len(infos):
                row = {
                    "episode": self.episode_num,
                    "reward": round(self._ep_rewards[i], 4),
                    "length": self._ep_lengths[i],
                    "xp": round(self._ep_xp[i], 4),
                    "kill": round(self._ep_kill[i], 4),
                    "death": round(infos[i].get("reward_components", {}).get("death", 0.0), 4),
                    "quest_proximity": round(self._ep_quest_prox[i], 4),
                    "quest_progress": round(infos[i].get("reward_components", {}).get("quest_progress", 0.0), 4),
                    "enemy_proximity": round(self._ep_enemy_prox[i], 4),
                    "target_acquired": round(self._ep_target_acq[i], 4),
                }
                for a in range(ACTION_COUNT):
                    row[f"act_{a}"] = int(self.current_actions[a])

                self.rows.append(row)
                self.episode_num += 1
                self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int32)
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_xp[i] = 0.0
                self._ep_kill[i] = 0.0
                self._ep_enemy_prox[i] = 0.0
                self._ep_target_acq[i] = 0.0
                self._ep_quest_prox[i] = 0.0

        return True

    def _on_rollout_end(self) -> None:
        self._write_csv()

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
    timesteps = int(os.environ.get("NEURALBOT_TIMESTEPS", "20000000"))
    model_path = os.environ.get("NEURALBOT_MODEL", "wow_neuralbot_model_v2")
    stats_path = model_path + "_episodes.csv"

    num_bots = len(BOT_NAMES)
    print(f"Creating ThreadedVecEnv with {num_bots} bots + SBX PPO (JAX)...")
    env = ThreadedVecEnv([make_env(name, host, port) for name in BOT_NAMES])
    obs = env.reset()

    # Ensure batch_size divides n_steps * n_envs evenly
    buffer_size = 256 * num_bots
    batch_size = 1024
    if buffer_size % batch_size != 0:
        print(f"Warning: buffer_size={buffer_size} not divisible by batch_size={batch_size}")

    print(f"Buffer: {buffer_size} samples, {buffer_size // batch_size} minibatches")
    print(f"Training PPO for {timesteps} timesteps using {num_bots} parallel envs...")
    print(f"Episode stats: {stats_path}")

    checkpoint_path = os.path.dirname(model_path) or "."
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=checkpoint_path,
        name_prefix="wow_neuralbot_v2",
    )
    stats_callback = EpisodeStatsCallback(stats_path)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
    )

    print(f"Starting model.learn() on device={model.device} ...", flush=True)
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, stats_callback],
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()


if __name__ == "__main__":
    main()
