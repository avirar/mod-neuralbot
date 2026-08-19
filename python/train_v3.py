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
from shared_memory_env import SharedMemoryVecEnv, OBS_FLAT_SIZE, FRAME_BYTES
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

    @property
    def INSERT_SQL(self):
        act_cols = ", ".join(f"act_{i}" for i in range(ACTION_COUNT))
        act_ph = ", ".join(["%s"] * ACTION_COUNT)
        return (
            f"INSERT INTO neuralbot_episodes "
            f"(episode, reward, length, xp, kill_count, death, "
            f" quest_proximity, quest_progress, enemy_proximity, target_acquired, "
            f" {act_cols}) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {act_ph})"
        )

    def __init__(self, db_config: dict, verbose=0):
        super().__init__(verbose)
        self.db_config = db_config
        self.episode_num = 0
        self.current_actions = np.zeros(ACTION_COUNT, dtype=np.int64)
        n = NUM_BOTS
        self._ep_rewards = np.zeros(n)
        self._ep_lengths = np.zeros(n, dtype=np.int64)
        self._ep_xp = np.zeros(n)
        self._ep_kill = np.zeros(n)
        self._ep_enemy_prox = np.zeros(n)
        self._ep_target_acq = np.zeros(n)
        self._ep_quest_prox = np.zeros(n)
        self._rollout_rows = []
        self._conn = None
        # reward-component column indices (REWARD_COMPONENT_KEYS order)
        from neuralbot_client import REWARD_COMPONENT_KEYS as _RCK
        self._rc_idx = {k: i for i, k in enumerate(_RCK)}

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

        # Vectorized accumulation over all envs (was a 400-iteration python loop/step).
        if len(actions):
            self.current_actions += np.bincount(
                np.asarray(actions, dtype=np.int64), minlength=ACTION_COUNT)
        if len(rewards):
            self._ep_rewards += np.asarray(rewards, dtype=np.float64)
        self._ep_lengths += 1

        env = self.model.get_env()
        comp = getattr(env, "last_components", None)
        if comp is not None:
            self._ep_xp         += comp[:, self._rc_idx["xp"]]
            self._ep_kill       += comp[:, self._rc_idx["kill"]]
            self._ep_enemy_prox += comp[:, self._rc_idx["enemy_proximity"]]
            self._ep_target_acq += comp[:, self._rc_idx["target_acquired"]]
            self._ep_quest_prox += comp[:, self._rc_idx["quest_proximity"]]

        for i in np.nonzero(dones)[0]:
            i = int(i)
            rc = infos[i].get("reward_components", {}) if (infos := self.locals.get("infos", [])) and i < len(infos) else {}
            row = (
                self.episode_num,
                round(float(self._ep_rewards[i]), 4),
                int(self._ep_lengths[i]),
                round(float(self._ep_xp[i]), 4),
                round(float(self._ep_kill[i]), 4),
                round(rc.get("death", 0.0), 4),
                round(float(self._ep_quest_prox[i]), 4),
                round(rc.get("quest_progress", 0.0), 4),
                round(float(self._ep_enemy_prox[i]), 4),
                round(float(self._ep_target_acq[i]), 4),
                *self.current_actions.tolist(),
            )
            self._rollout_rows.append(row)
            self.episode_num += 1
            self.current_actions[:] = 0
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
    learning_rate = float(os.environ.get("NEURALBOT_LR", "4e-4"))
    ent_coef = float(os.environ.get("NEURALBOT_ENT", "0.01"))
    n_steps_env = int(os.environ.get("NEURALBOT_NSTEPS", "1024"))
    reward_clip = float(os.environ.get("NEURALBOT_REWARD_CLIP", "0.3"))
    target_kl = float(os.environ.get("NEURALBOT_KL", "0.008"))
    model_path = os.environ.get("NEURALBOT_MODEL", "wow_neuralbot_model_v3")
    model_zip = f"{model_path}.zip"
    has_previous = os.path.exists(model_zip)

    # When resuming, save to a new path to preserve previous iteration
    save_path = f"{model_path}_iter2" if has_previous else model_path

    print(f"Connecting to shared memory at {os.environ.get('SHM_PATH', '/dev/shm/neuralbot_shm')}...")
    env = SharedMemoryVecEnv(timeout=60.0)

    num_bots = env.num_envs
    buffer_size = 512 * num_bots
    batch_size = 1024

    print(f"SharedMemoryVecEnv ready: {num_bots} bots, {OBS_FLAT_SIZE} obs dims, {FRAME_BYTES} bytes/frame")
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
            model = PPO.load(model_path, env=env, custom_objects={"n_steps": n_steps_env})
            model.learning_rate = learning_rate  # allow NEURALBOT_LR to retune resumed runs
            model.ent_coef = ent_coef          # and NEURALBOT_ENT (0.005 over-commits and oscillates)
            model.target_kl = target_kl        # and NEURALBOT_KL (0.02 permits mode-dropping drift)
            print(f"Model loaded from {model_zip} (lr={learning_rate}). Will save final as {save_path}.zip", flush=True)
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
            n_steps=n_steps_env,
            batch_size=batch_size,
            learning_rate=learning_rate,
            # Sparse delayed reward: a kill needs ~190 steps of approach before combat,
            # so the discount horizon must cover it (0.99 -> ~100 steps was too short;
            # 0.999 -> ~1000 steps puts the XP inside the credit window).
            gamma=0.999,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=ent_coef,
            # KL guard: early-stop each rollout's epochs when the policy lurches
            # (v8 @ lr 7e-4 collapsed explained_variance 0.94 -> 0.009 with
            # clip_fraction 0.13 — advantage signal turned to noise).
            target_kl=target_kl,
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
