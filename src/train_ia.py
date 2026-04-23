import os
import random
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

try:
    from src.env.pokemon_env import PokemonEnv
    from src.model_compat import save_model_metadata
except ImportError:
    from env.pokemon_env import PokemonEnv
    from model_compat import save_model_metadata


SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BattleMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.win_history = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward" in info:
                self.logger.record("battle/step_reward", float(info["reward"]))
            if "damage_dealt" in info:
                self.logger.record("battle/damage_dealt_running", float(info["damage_dealt"]))
            if "damage_received" in info:
                self.logger.record("battle/damage_received_running", float(info["damage_received"]))
            if "episode_reward" in info:
                self.win_history.append(1.0 if info.get("is_win") else 0.0)
                self.logger.record("battle/episode_reward", float(info["episode_reward"]))
                self.logger.record("battle/episode_length", float(info["episode_length"]))
                self.logger.record("battle/episode_damage_dealt", float(info["damage_dealt"]))
                self.logger.record("battle/episode_damage_received", float(info["damage_received"]))

        if self.win_history:
            self.logger.record("battle/win_rate_100", float(np.mean(self.win_history[-100:])))
        return True


def build_env(opponent_mode="random", opponent_model=None):
    env = PokemonEnv()
    env.reset(seed=SEED)
    env.set_opponent(mode=opponent_mode, model=opponent_model)
    return Monitor(env)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=150000)
    parser.add_argument("--load_model", type=str, default=None)
    return parser.parse_args()


def train(args):
    seed_everything(SEED)
    os.makedirs("models/best_model_s3", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = build_env(opponent_mode="random")
    eval_env = build_env(opponent_mode="random")

    # -------------------------
    # LOAD OR CREATE MODEL
    # -------------------------
    if args.load_model:
        print(f"\nLoading model from {args.load_model}\n")
        model = PPO.load(args.load_model, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            ent_coef=0.02,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="auto",
            seed=SEED,
        )

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path="./models/best_model_s3/",
                log_path="./logs/",
                eval_freq=5000,
                n_eval_episodes=25,
                deterministic=True,
            ),
            BattleMetricsCallback(),
        ]
    )

    print("\nStarting PPO training...\n")
    print("Total timesteps:", args.total_timesteps)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False if args.load_model else True
    )

    final_model_path = "models/pokemon_ia_v5_avanzada"
    model.save(final_model_path)
    save_model_metadata(final_model_path)

    best_model_path = "models/best_model_s3/best_model"
    if os.path.exists(best_model_path + ".zip"):
        save_model_metadata(best_model_path)

    print("\nTraining complete. Saved model to models/pokemon_ia_v5_avanzada.zip")


if __name__ == "__main__":
    args = get_args()
    train(args)