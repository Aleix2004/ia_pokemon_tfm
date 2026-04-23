import os
import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch

try:
    from src.env.pokemon_env import PokemonEnv
    from src.model_compat import check_model_compatibility, require_compatible_model, save_model_metadata
except ImportError:
    from env.pokemon_env import PokemonEnv
    from model_compat import check_model_compatibility, require_compatible_model, save_model_metadata


SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(mode="random", model=None):
    env = PokemonEnv()
    env.reset(seed=SEED)
    env.set_opponent(mode=mode, model=model, random_baseline_chance=0.5)
    return Monitor(env)


def evaluate(model, opponent_mode="random", opponent_model=None, episodes=40):
    env = PokemonEnv()
    env.set_opponent(mode=opponent_mode, model=opponent_model, random_baseline_chance=0.5)

    wins = 0
    rewards = []
    lengths = []
    dealt = []
    received = []
    ko_counts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
        wins += 1 if info.get("is_win") else 0
        rewards.append(info.get("episode_reward", 0.0))
        lengths.append(info.get("episode_length", 0))
        dealt.append(info.get("damage_dealt", 0.0))
        received.append(info.get("damage_received", 0.0))
        ko_counts.append(info.get("ko_count", 0))

    return {
        "win_rate": wins / episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths)),
        "avg_damage_dealt": float(np.mean(dealt)),
        "avg_damage_received": float(np.mean(received)),
        "avg_ko_count": float(np.mean(ko_counts)),
    }


def train():
    seed_everything(SEED)
    os.makedirs("models/self_play_history", exist_ok=True)
    os.makedirs("models/best_self_play", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = build_env(mode="random")
    seed_candidates = [
        "models/canonical_ppo_v1",
        "models/best_model_s3/best_model",
    ]
    model_path = next((path for path in seed_candidates if os.path.exists(path + ".zip")), None)

    if model_path is not None:
        compat = check_model_compatibility(model_path)
        if not compat.is_valid:
            raise RuntimeError(
                f"Seed model '{model_path}' is LEGACY - INCOMPATIBLE ({compat.reason}). "
                "Train a canonical model first (obs28/action4)."
            )
        print(f"Loading seed model from {model_path}.zip")
        model = PPO.load(model_path, env=env, tensorboard_log="./logs/", seed=SEED)
    else:
        print("No seed model found. Starting self-play from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.02,
            tensorboard_log="./logs/",
            seed=SEED,
        )

    frozen_opponent = None
    snapshot_path = None

    for generation in range(1, 11):
        if frozen_opponent is None:
            model.get_env().env_method("set_opponent", mode="random", model=None)
            phase = "random-baseline"
        else:
            model.get_env().env_method("set_opponent", mode="mixed", model=frozen_opponent, random_baseline_chance=0.5)
            phase = "mixed-self-play"

        print(f"\nGeneration {generation}: training against {phase}")
        model.learn(total_timesteps=25000, reset_num_timesteps=False, progress_bar=True)

        snapshot_path = os.path.join("models/self_play_history", f"gen_{generation}")
        model.save(snapshot_path)
        save_model_metadata(snapshot_path)
        frozen_opponent = require_compatible_model(snapshot_path)

        random_metrics = evaluate(model, opponent_mode="random", episodes=30)
        greedy_metrics = evaluate(model, opponent_mode="greedy", episodes=30)
        mirror_metrics = evaluate(model, opponent_mode="model", opponent_model=frozen_opponent, episodes=30)

        print(
            "Random WR: {0:.2%} | Greedy WR: {1:.2%} | Mirror WR: {2:.2%} | Avg Reward: {3:.3f}".format(
                random_metrics["win_rate"],
                greedy_metrics["win_rate"],
                mirror_metrics["win_rate"],
                random_metrics["avg_reward"],
            )
        )

    final_path = "models/best_self_play/model_final_v4"
    model.save(final_path)
    save_model_metadata(final_path)
    print(f"\nSelf-play training complete. Saved final model to {final_path}.zip")


if __name__ == "__main__":
    train()
