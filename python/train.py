#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from wow_neuralbot_env import WoWNeuralBotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np


def main():
    host = os.environ.get("NEURALBOT_HOST", "127.0.0.1")
    port = int(os.environ.get("NEURALBOT_PORT", "9000"))
    timesteps = int(os.environ.get("NEURALBOT_TIMESTEPS", "1000000"))
    model_path = os.environ.get("NEURALBOT_MODEL", "wow_neuralbot_model")

    print(f"Connecting to NeuralBot server at {host}:{port}...")
    env = WoWNeuralBotEnv(host=host, port=port)
    env.client.connect()

    status = env.client.status()
    print(f"Server status: {status}")

    if status.get("state") != "READY":
        print("ERROR: Bot is not ready. Ensure the character is logged in.")
        env.close()
        sys.exit(1)

    spellbook = env.client.get_spellbook()
    if not spellbook:
        print("WARNING: No spells found in bot's spellbook.")
    else:
        print(f"Discovered {len(spellbook)} spells in spellbook: {spellbook}")

    print(f"Training PPO for {timesteps} timesteps...")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="wow_neuralbot",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()


if __name__ == "__main__":
    main()
