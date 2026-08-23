#!/usr/bin/env python3
"""
Behavior-cloning warm-start: train the SBX PPO policy's action head to imitate expert
playerbot actions, then save it for PPO fine-tuning by train_v3.py.

Pipeline:
  1. Read demos.bin (recorded (frame, actionName, targetGuid) records).
  2. Map each record to the 41-action space via bc_mapping (skip meta/noise).
  3. Supervised cross-entropy on the policy's action distribution (JAX, SBX's own net).
  4. Save <out>.zip — train_v3.py loads it with PPO.load and fine-tunes.

Usage:
    python3 bc_train.py python/bc_demos/demos.bin --out wow_neuralbot_model_v14_bc.zip
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from neuralbot_client import ACTION_COUNT, FRAME_DTYPE
from neuralbot_shm import flatten_frames, OBS_FLAT_SIZE
from bc_analyze import load_records
from bc_mapping import build_mapper

DEFAULT_DBC = "/home/luke/GIT/azerothcore-wotlk/env/dist/bin/dbc/Spell.dbc"


def make_dummy_env():
    """Dummy env with the right spaces — only used to build the SBX policy network."""
    def _f():
        class Dummy(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-10.0, 10.0, (OBS_FLAT_SIZE,), np.float32)
                self.action_space = gym.spaces.Discrete(ACTION_COUNT)

            def reset(self, *a, **k):
                return np.zeros(OBS_FLAT_SIZE, np.float32), {}

            def step(self, action):
                return np.zeros(OBS_FLAT_SIZE, np.float32), 0.0, False, False, {}

        return Dummy()

    return DummyVecEnv([_f])


def build_dataset(records, mapper):
    frames, actions = [], []
    for i in range(len(records)):
        name = records["name"][i].decode("utf-8", "replace").rstrip("\x00")
        frame = records["frame"][i]
        aid = mapper(frame, name, int(records["targetGuid"][i]))
        if aid is None:
            continue
        frames.append(frame)
        actions.append(aid)

    if not frames:
        return None, None
    X = flatten_frames(np.array(frames, dtype=FRAME_DTYPE))
    Y = np.asarray(actions, dtype=np.int64)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demos", help="path to demos.bin")
    ap.add_argument("--spell-dbc", default=DEFAULT_DBC)
    ap.add_argument("--out", default="wow_neuralbot_model_v14_bc.zip")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="cap records read (0 = all)")
    ap.add_argument("--net-arch", default="256,256")
    args = ap.parse_args()

    records = load_records(args.demos, args.limit)
    if len(records) == 0:
        print(f"no records in {args.demos}")
        return
    mapper = build_mapper(args.spell_dbc)

    X, Y = build_dataset(records, mapper)
    if X is None:
        print("no mappable records")
        return
    print(f"BC dataset: {len(Y)} samples from {len(records)} records")
    dist = {int(i): int(c) for i, c in enumerate(np.bincount(Y, minlength=ACTION_COUNT)) if c > 0}
    print("action distribution:", dist)

    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    env = make_dummy_env()
    model_kwargs = dict(
        policy="MlpPolicy", env=env, n_steps=1024, batch_size=1024,
        learning_rate=1e-4, n_epochs=10, gamma=0.999, gae_lambda=0.98,
        clip_range=0.2, ent_coef=0.01, target_kl=None, verbose=0,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=1,  # save with verbose=1 so PPO.load() logs metrics after warm-start
    )
    from sbx import PPO
    model = PPO(**model_kwargs)

    actor_state = model.policy.actor_state

    def bc_loss(params, obs, labels):
        dist = actor_state.apply_fn(params, obs)
        return -dist.log_prob(labels).mean()

    grad_fn = jax.jit(jax.value_and_grad(bc_loss))

    rng = np.random.default_rng(0)
    n = len(Y)
    Xj = jnp.asarray(X)
    Yj = jnp.asarray(Y)

    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        losses = []
        for b in range(0, n, args.batch_size):
            idx = perm[b:b + args.batch_size]
            loss, grads = grad_fn(actor_state.params, Xj[idx], Yj[idx])
            actor_state = actor_state.apply_gradients(grads=grads)
            losses.append(float(loss))
        print(f"epoch {epoch:3d}: bc_loss {np.mean(losses):.4f}", flush=True)

    # Hand the trained actor back to the model and save for PPO.load().
    model.policy.actor_state = actor_state
    if hasattr(model, "actor"):
        model.actor = actor_state
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
