"""
main.py
~~~~~~~
Entry point for the Pokémon IA training pipeline.

Delegates to ``train_ppo.py``, which is the production training script:
SubprocVecEnv × N workers, VecNormalize, WinRateCallback, and full
checkpoint management.  Use ``train_ia.py`` only for lightweight
single-env experiments or quick smoke-tests.

Usage
-----
    # Standard training run (all defaults from train_ppo.py CLI)
    python -m src.main

    # Pass arguments through to train_ppo
    python -m src.train_ppo --n_envs 16 --total_timesteps 1_000_000
"""
from __future__ import annotations

try:
    from src.train_ppo import main as train_main
except ImportError:
    from train_ppo import main as train_main


if __name__ == "__main__":
    train_main()
