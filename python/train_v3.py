#!/usr/bin/env python3
"""
Training script v3: SBX PPO (JAX) + SharedMemoryVecEnv — zero-copy IPC.

Eliminates 400 TCP sockets and 400 Python threads. Single-threaded
shared memory batch protocol for ~2-3x throughput.

Episode stats stored in MySQL (acore_characters.neuralbot_episodes) —
append-only, no file rewriting, no degradation.
"""
import sys
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
import numpy as np
import pymysql

sys.path.insert(0, os.path.dirname(__file__))

from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from shared_memory_env import SharedMemoryVecEnv, SHM_OBS_PER_BOT
from neuralbot_client import ACTION_COUNT, NUM_BOTS

log = logging.getLogger("train_v3")

DB_CONFIG = {
    "host": os.environ.get("NEURALBOT_DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("NEURALBOT_DB_PORT", "3306")),
    "user": os.environ.get("NEURALBOT_DB_USER", "acore"),
    "password": os.environ.get("NEURALBOT_DB_PASS", "abc"),
    "database": os.environ.get("NEURALBOT_DB_NAME", "acore_characters"),
}


class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode stats to MySQL. Batched INSERT on each rollout end."""

    INSERT_SQL = (
        "INSERT INTO neuralbot_episodes "
        "(episode, reward, length, xp, kill_count, death, "
        " quest_proximity, quest_progress, enemy_proximity, target_acquired, "
        " act_0, act_1, act_2, act_3, act_4, act_5, act_6, act_7, "
        " act_8, act_9, act_10, act_11, act_12, act_13, act_14) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
        " %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )

    def __init__(self, db_config: dict, verbose=0):
        super().__init__(verbose)
        self.db_config = db_config
        self.episode_num = 0
        self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int32)
        self._ep_rewards = [0.0] * NUM_BOTS
        self._ep_lengths = [0] * NUM_BOTS
        self._ep_xp = [0.0] * NUM_BOTS
        self._ep_kill = [0.0] * NUM_BOTS
        self._ep_enemy_prox = [0.0] * NUM_BOTS
        self._ep_target_acq = [0.0] * NUM_BOTS
        self._ep_quest_prox = [0.0] * NUM_BOTS
        self._rollout_rows = []
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = pymysql.connect(**self.db_config)
            self._conn.autocommit(True)
        else:
            self._conn.ping(reconnect=True)
        return self._conn

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
                self._ep_xp[i]         += rc.get("xp", 0.0)
                self._ep_kill[i]       += rc.get("kill", 0.0)
                self._ep_enemy_prox[i] += rc.get("enemy_proximity", 0.0)
                self._ep_target_acq[i] += rc.get("target_acquired", 0.0)
                self._ep_quest_prox[i] += rc.get("quest_proximity", 0.0)

            if dones[i] and i < len(infos):
                row = (
                    self.episode_num,
                    round(self._ep_rewards[i], 4),
                    self._ep_lengths[i],
                    round(self._ep_xp[i], 4),
                    round(self._ep_kill[i], 4),
                    round(infos[i].get("reward_components", {}).get("death", 0.0), 4),
                    round(self._ep_quest_prox[i], 4),
                    round(infos[i].get("reward_components", {}).get("quest_progress", 0.0), 4),
                    round(self._ep_enemy_prox[i], 4),
                    round(self._ep_target_acq[i], 4),
                    int(self.current_actions[0]),  int(self.current_actions[1]),
                    int(self.current_actions[2]),  int(self.current_actions[3]),
                    int(self.current_actions[4]),  int(self.current_actions[5]),
                    int(self.current_actions[6]),  int(self.current_actions[7]),
                    int(self.current_actions[8]),  int(self.current_actions[9]),
                    int(self.current_actions[10]), int(self.current_actions[11]),
                    int(self.current_actions[12]), int(self.current_actions[13]),
                    int(self.current_actions[14]),
                )
                self._rollout_rows.append(row)
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
        self._flush_to_mysql()

    def _flush_to_mysql(self):
        if not self._rollout_rows:
            return
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.executemany(self.INSERT_SQL, self._rollout_rows)
            self._rollout_rows.clear()
        except Exception as e:
            print(f"[stats] MySQL write error: {e}", flush=True)
            # Don't lose data — keep in memory for next attempt
            try:
                self._conn = None  # force reconnect next time
            except Exception:
                pass


def main():
    host = os.environ.get("NEURALBOT_HOST", "127.0.0.1")
    port = int(os.environ.get("NEURALBOT_PORT", "9000"))
    timesteps = int(os.environ.get("NEURALBOT_TIMESTEPS", "20000000"))
    model_path = os.environ.get("NEURALBOT_MODEL", "wow_neuralbot_model_v3")
    model_zip = f"{model_path}.zip"
    has_previous = os.path.exists(model_zip)

    # When resuming, save to a new path to preserve previous iteration
    save_path = f"{model_path}_iter2" if has_previous else model_path

    print(f"Connecting to shared memory at {os.environ.get('SHM_PATH', '/dev/shm/neuralbot_shm')}...")
    env = SharedMemoryVecEnv(timeout=60.0)

    num_bots = env.num_envs
    buffer_size = 256 * num_bots
    batch_size = 1024

    print(f"SharedMemoryVecEnv ready: {num_bots} bots, {SHM_OBS_PER_BOT} floats/bot")
    print(f"Buffer: {buffer_size} samples, {buffer_size // batch_size} minibatches")
    print(f"Training PPO for {timesteps} timesteps using {num_bots} parallel envs...")
    print(f"Episode stats → MySQL {DB_CONFIG['host']}/{DB_CONFIG['database']}.neuralbot_episodes", flush=True)

    checkpoint_path = os.path.dirname(save_path) or "."
    # save_freq=2500 means save every 2500 rollout steps = 2500 × 400envs = 1M global steps
    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=checkpoint_path,
        name_prefix=os.path.basename(save_path),
    )
    stats_callback = EpisodeStatsCallback(DB_CONFIG)

    if has_previous:
        try:
            model = PPO.load(model_path, env=env)
            print(f"Model loaded from {model_zip}. Will save final as {save_path}.zip", flush=True)
        except Exception as e:
            print(f"Model load failed: {e}", flush=True)
            print("Creating fresh model (action/obs space may have changed).", flush=True)
            has_previous = False
            save_path = model_path
    if not has_previous:
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
            ent_coef=0.05,
        )
        print(f"Fresh model created.", flush=True)

    print(f"Starting model.learn() on device={model.device} ...", flush=True)
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, stats_callback],
    )

    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Close stats connection
    if stats_callback._conn:
        stats_callback._conn.close()

    env.close()


if __name__ == "__main__":
    main()
