"""
self_play.py
~~~~~~~~~~~~
AlphaZero-inspired self-play training system for the Pokémon RL environment.

ARCHITECTURE OVERVIEW
─────────────────────

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    SELF-PLAY TRAINING LOOP                          │
  │                                                                      │
  │  ┌──────────────┐   sample    ┌──────────────────────────────────┐  │
  │  │ OpponentPool │ ──70/30──►  │  VecEnv workers (N processes)   │  │
  │  │              │             │  each env has one frozen opponent │  │
  │  │ ● v1 ←oldest │             └──────────┬───────────────────────┘  │
  │  │ ● v2         │                        │ episodes (obs, act, rew) │
  │  │ ● v3         │                        ▼                          │
  │  │ ● v4 ←latest │             ┌──────────────────────────────────┐  │
  │  └──────────────┘             │   PPO update (SB3 learn loop)    │  │
  │         ▲                     └──────────┬───────────────────────┘  │
  │         │ add new version                │ new policy weights        │
  │         │                               ▼                           │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │  Generation end:                                             │   │
  │  │  1. Save checkpoint  model_v{N}.zip                         │   │
  │  │  2. Evaluate vs pool (all versions) → win matrix             │   │
  │  │  3. Update Elo ratings                                       │   │
  │  │  4. Add checkpoint to OpponentPool (circular, max_size=10)   │   │
  │  │  5. Print leaderboard                                        │   │
  │  └──────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────┘

SAMPLING STRATEGY
─────────────────
  70%  →  latest model in pool (current best, strongest signal)
  30%  →  uniform random from older pool entries (diversity)

  Rationale: always training against the latest version creates a cyclic
  best-response problem.  The 30% historical opponents provide a broader
  coverage of the strategy space, similar to OpenAI Five's past-self mixing.

ELO SYSTEM
──────────
  Standard Elo with K=32.  Each model version is seeded at 1500.
  After each generation's evaluation, Elo is updated for every
  (current_model, pool_member) pair using the win-rate as the score.

USAGE
─────
    # Default: 20 generations, 50 k timesteps/gen, 8 parallel envs
    python -m src.self_play

    # Custom
    python -m src.self_play --generations 30 --timesteps_per_gen 100000 \\
                            --n_envs 16 --max_pool_size 15

    # Continue from a checkpoint
    python -m src.self_play --load_model models/self_play_v2/gen_010 \\
                            --load_vecnorm models/self_play_v2/gen_010_vecnorm.pkl

OUTPUT
──────
    models/self_play_v2/gen_{N:03d}.zip           — generation checkpoint
    models/self_play_v2/gen_{N:03d}_vecnorm.pkl   — matching VecNorm stats
    models/self_play_v2/elo_ratings.json          — Elo ratings per version
    models/self_play_v2/eval_log.csv              — full evaluation log
    logs/self_play_*/                             — TensorBoard logs
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

try:
    from src.env.pokemon_env import PokemonEnv
    from src.model_compat import save_model_metadata
    from src.train_ppo import (
        WinRateCallback,
        build_eval_env,
        build_train_env,
        evaluate,
        make_env,
        print_eval_results,
        seed_everything,
    )
except ImportError:
    from env.pokemon_env import PokemonEnv
    from model_compat import save_model_metadata
    from train_ppo import (
        WinRateCallback,
        build_eval_env,
        build_train_env,
        evaluate,
        make_env,
        print_eval_results,
        seed_everything,
    )


SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  OPPONENT POOL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoolEntry:
    """A single entry in the opponent pool."""
    version:     int          # generation index
    model_path:  str          # path without .zip extension
    vecnorm_path: str | None  # matching VecNorm .pkl (can be None)


class OpponentPool:
    """
    Circular buffer of historical model checkpoints.

    Entries are appended in generation order.  When the pool exceeds
    max_size, the oldest entry is evicted (deque behaviour).

    Sampling
    ────────
    sample(p_latest=0.70) returns a PoolEntry:
      • With probability p_latest  → most recently added entry
      • With probability 1−p_latest → uniformly random from all entries

    This means the current best opponent dominates, while past versions
    provide diversity to prevent cyclic best-response collapse.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._pool: Deque[PoolEntry] = collections.deque(maxlen=max_size)

    def add(self, entry: PoolEntry) -> None:
        """Add a new checkpoint to the pool."""
        self._pool.append(entry)

    @property
    def latest(self) -> PoolEntry | None:
        """Return the most recently added entry, or None if pool is empty."""
        return self._pool[-1] if self._pool else None

    @property
    def size(self) -> int:
        return len(self._pool)

    def sample(
        self,
        p_latest:     float = 0.70,
        rng:          random.Random | None = None,
    ) -> PoolEntry | None:
        """
        Sample one opponent from the pool.

        Parameters
        ----------
        p_latest : float
            Probability of returning the latest (strongest) opponent.
            Remaining probability is split uniformly over all entries.
        rng : random.Random | None
            Optional seeded RNG for reproducibility.

        Returns
        -------
        PoolEntry or None if pool is empty.
        """
        if not self._pool:
            return None
        if len(self._pool) == 1:
            return self._pool[-1]

        _rng = rng or random
        if _rng.random() < p_latest:
            return self._pool[-1]                     # latest
        return _rng.choice(list(self._pool)[:-1])     # random older entry

    def all_entries(self) -> list[PoolEntry]:
        """Return all pool entries from oldest to newest."""
        return list(self._pool)

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        entries = ", ".join(f"v{e.version}" for e in self._pool)
        return f"OpponentPool([{entries}], max={self.max_size})"


# ─────────────────────────────────────────────────────────────────────────────
#  ELO RATING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class EloRating:
    """
    Standard Elo rating system for tracking relative strength between versions.

    Each model version is identified by its version integer.  Ratings start
    at DEFAULT_RATING (1500) and are updated after every evaluation matchup.

    Update formula
    ──────────────
        expected_A = 1 / (1 + 10^((R_B − R_A) / 400))
        R_A_new    = R_A + K × (score_A − expected_A)

    where score_A is the win-rate of A against B (continuous in [0, 1]).
    Using win-rate instead of binary outcome averages across all evaluation
    episodes, giving a smoother Elo update.

    Parameters
    ----------
    k_factor : float
        Elo K-factor.  Higher = faster rating change per match.
        Standard chess: K=16 (stable) / K=32 (active).  Use K=32 here
        because early training has high variance.
    default_rating : float
        Starting Elo for new versions.  1500 is conventional.
    """

    def __init__(self, k_factor: float = 32.0, default_rating: float = 1500.0):
        self.k_factor       = k_factor
        self.default_rating = default_rating
        self._ratings: Dict[int, float] = {}
        self._match_history: List[dict] = []

    def get_rating(self, version: int) -> float:
        """Return the current Elo rating for a version (seeded at default if new)."""
        return self._ratings.get(version, self.default_rating)

    def update(
        self,
        version_a:  int,
        version_b:  int,
        win_rate_a: float,
    ) -> Tuple[float, float]:
        """
        Update ratings for a matchup where A achieved win_rate_a against B.

        Parameters
        ----------
        version_a  : integer version identifier for the active model
        version_b  : integer version identifier for the opponent
        win_rate_a : float in [0, 1] — win rate of A in this matchup

        Returns
        -------
        (new_rating_a, new_rating_b)
        """
        ra = self.get_rating(version_a)
        rb = self.get_rating(version_b)

        expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        expected_b = 1.0 - expected_a

        score_a = float(win_rate_a)
        score_b = 1.0 - score_a

        new_ra = ra + self.k_factor * (score_a - expected_a)
        new_rb = rb + self.k_factor * (score_b - expected_b)

        self._ratings[version_a] = new_ra
        self._ratings[version_b] = new_rb

        self._match_history.append({
            "version_a": version_a, "version_b": version_b,
            "win_rate_a": score_a,  "expected_a": expected_a,
            "delta_a": new_ra - ra, "new_ra": new_ra, "new_rb": new_rb,
        })
        return new_ra, new_rb

    def leaderboard(self) -> list[tuple[int, float]]:
        """Return sorted list of (version, rating) descending by rating."""
        return sorted(self._ratings.items(), key=lambda x: -x[1])

    def print_leaderboard(self) -> None:
        """Print a formatted Elo leaderboard to stdout."""
        board = self.leaderboard()
        if not board:
            print("  (no ratings yet)")
            return
        print(f"\n  {'Rank':<6} {'Version':<10} {'Elo Rating':>12}")
        print(f"  {'─'*6} {'─'*10} {'─'*12}")
        for rank, (ver, rating) in enumerate(board, 1):
            print(f"  {rank:<6} v{ver:<9} {rating:>12.1f}")

    def save(self, path: str) -> None:
        """Save ratings to a JSON file."""
        data = {
            "ratings":       {str(k): v for k, v in self._ratings.items()},
            "k_factor":      self.k_factor,
            "default_rating":self.default_rating,
            "match_count":   len(self._match_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EloRating":
        """Load ratings from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        elo = cls(k_factor=data.get("k_factor", 32.0),
                  default_rating=data.get("default_rating", 1500.0))
        elo._ratings = {int(k): v for k, v in data.get("ratings", {}).items()}
        return elo


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION LOG
# ─────────────────────────────────────────────────────────────────────────────

class EvalLog:
    """
    Append-only CSV log of all evaluation matchups across generations.

    Columns
    ───────
    generation, current_version, opponent_version, opponent_mode,
    win_rate, avg_reward, avg_length, elo_current, elo_opponent
    """

    COLUMNS = [
        "generation", "current_version", "opponent_version", "opponent_mode",
        "win_rate", "avg_reward", "avg_length",
        "avg_damage_dealt", "avg_damage_received",
        "elo_current", "elo_opponent",
    ]

    def __init__(self, path: str):
        self.path = path
        self._is_new = not os.path.exists(path)

    def append(self, row: dict) -> None:
        """Append a single evaluation result row."""
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            if self._is_new:
                writer.writeheader()
                self._is_new = False
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATION CALLBACK  (fires inside PPO.learn())
# ─────────────────────────────────────────────────────────────────────────────

class GenerationProgressCallback(WinRateCallback):
    """
    Lightweight callback for within-generation metrics logging.

    Inherits WinRateCallback for win-rate tracking; adds generation-level
    context to TensorBoard log keys so curves from different generations
    are distinguishable.
    """

    def __init__(self, generation: int, **kwargs):
        super().__init__(**kwargs)
        self.generation = generation

    def _on_step(self) -> bool:
        ok = super()._on_step()
        # Log generation index so TensorBoard can colour-code by generation
        self.logger.record("self_play/generation", self.generation)
        return ok


# ─────────────────────────────────────────────────────────────────────────────
#  ENV BUILDER WITH OPPONENT POOL
# ─────────────────────────────────────────────────────────────────────────────

def build_self_play_env(
    pool:              OpponentPool,
    n_envs:            int,
    seed:              int          = SEED,
    p_latest:          float        = 0.70,
    norm_obs:          bool         = True,
    norm_reward:       bool         = True,
    clip_obs:          float        = 10.0,
    gamma:             float        = 0.99,
    rng:               random.Random | None = None,
    registry_snapshot: bytes | None = None,
) -> VecNormalize:
    """
    Build a SubprocVecEnv where each worker plays against an opponent sampled
    from the pool.

    If the pool is empty, workers use the "random" baseline opponent.
    Otherwise, each of the N workers independently samples from the pool
    using the 70/30 strategy, producing a diverse opponent distribution
    across the batch.

    Parameters
    ----------
    pool      : OpponentPool with PoolEntry objects
    n_envs    : number of parallel workers
    p_latest  : probability of sampling the latest opponent per worker
    rng       : seeded Python RNG for reproducible sampling

    Notes on model loading
    ──────────────────────
    Each worker needs its own frozen copy of the opponent model.  PPO.load()
    returns a new model instance per call, so N workers get N independent
    copies (no shared state).  VecNormalize is NOT applied to the opponent's
    observation — the opponent receives raw obs (identical to training env
    obs before normalisation), which is consistent with how frozen models
    are used in the existing self-play code.
    """
    def _make_worker(rank: int) -> callable:
        """Return a thunk for worker #rank with its sampled opponent."""
        entry = pool.sample(p_latest=p_latest, rng=rng)

        if entry is None:
            # Pool empty → pure random opponent
            return make_env(rank=rank, seed=seed, opponent_mode="random",
                            registry_snapshot=registry_snapshot)

        # Load frozen opponent model
        try:
            frozen = PPO.load(entry.model_path)
            mode   = "model"
        except Exception as e:
            print(f"[SelfPlay] Warning: could not load {entry.model_path}: {e}. Using random.")
            frozen = None
            mode   = "random"

        return make_env(
            rank               = rank,
            seed               = seed,
            opponent_mode      = mode,
            opponent_model     = frozen,
            registry_snapshot  = registry_snapshot,
        )

    _start_method = "spawn" if sys.platform == "win32" else "fork"
    envs = SubprocVecEnv(
        [_make_worker(i) for i in range(n_envs)],
        start_method=_start_method,
    )
    venv = VecNormalize(
        envs,
        norm_obs    = norm_obs,
        norm_reward = norm_reward,
        clip_obs    = clip_obs,
        gamma       = gamma,
        training    = True,
    )
    return venv


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION AGAINST POOL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vs_pool(
    model,
    pool:         OpponentPool,
    elo:          EloRating,
    eval_log:     EvalLog,
    current_ver:  int,
    generation:   int,
    episodes:     int   = 50,
    vecnorm_path: str | None = None,
    seed:         int   = SEED,
) -> dict:
    """
    Evaluate the current model against every entry in the pool and update Elo.

    For each pool entry:
      1. Load the frozen opponent
      2. Run `episodes` evaluation episodes
      3. Update Elo ratings (current_ver vs pool_entry.version)
      4. Log result row to eval_log

    Returns a summary dict:
      {pool_entry.version: {win_rate, avg_reward, ...}, ...}
    """
    results = {}

    for entry in pool.all_entries():
        try:
            opponent = PPO.load(entry.model_path)
            mode     = "model"
        except Exception as e:
            print(f"  [Eval] Cannot load {entry.model_path}: {e}")
            opponent = None
            mode     = "random"

        metrics = evaluate(
            model          = model,
            vecnorm_path   = vecnorm_path,
            opponent_mode  = mode,
            opponent_model = opponent,
            episodes       = episodes,
            seed           = seed + entry.version,
            deterministic  = True,
        )
        results[entry.version] = metrics

        # Update Elo
        new_ra, new_rb = elo.update(
            version_a  = current_ver,
            version_b  = entry.version,
            win_rate_a = metrics["win_rate"],
        )

        eval_log.append({
            "generation":       generation,
            "current_version":  current_ver,
            "opponent_version": entry.version,
            "opponent_mode":    mode,
            "win_rate":         metrics["win_rate"],
            "avg_reward":       metrics["avg_reward"],
            "avg_length":       metrics["avg_length"],
            "avg_damage_dealt": metrics["avg_damage_dealt"],
            "avg_damage_received": metrics["avg_damage_received"],
            "elo_current":      new_ra,
            "elo_opponent":     new_rb,
        })

        print(
            f"  vs v{entry.version:<4}  WR={metrics['win_rate']:.1%}  "
            f"R={metrics['avg_reward']:+.3f}  "
            f"Elo: {new_ra:.0f}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ARGS
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaZero-style self-play for the Pokémon RL system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loop control
    p.add_argument("--generations",        type=int,   default=20)
    p.add_argument("--timesteps_per_gen",  type=int,   default=50_000)
    p.add_argument("--n_envs",             type=int,   default=8)
    p.add_argument("--seed",               type=int,   default=SEED)
    p.add_argument("--run_name",           type=str,   default="self_play_v2")

    # Pool & sampling
    p.add_argument("--max_pool_size",      type=int,   default=10)
    p.add_argument("--p_latest",           type=float, default=0.70,
                   help="Probability of sampling the latest opponent (0..1)")
    p.add_argument("--random_warmup_gens", type=int,   default=2,
                   help="Initial generations with random opponent (before using pool)")

    # PPO hyperparameters (forwarded to model init)
    p.add_argument("--lr",                 type=float, default=2.5e-4)
    p.add_argument("--n_steps",            type=int,   default=2048)
    p.add_argument("--batch_size",         type=int,   default=128)
    p.add_argument("--ent_coef",           type=float, default=0.02)
    p.add_argument("--gamma",              type=float, default=0.99)
    p.add_argument("--gae_lambda",         type=float, default=0.95)

    # VecNormalize
    p.add_argument("--no_norm_obs",        action="store_true")
    p.add_argument("--no_norm_reward",     action="store_true")
    p.add_argument("--clip_obs",           type=float, default=10.0)

    # Evaluation
    p.add_argument("--eval_episodes",      type=int,   default=50)
    p.add_argument("--eval_vs_random",     action="store_true", default=True,
                   help="Also evaluate vs random baseline each generation")

    # Checkpointing / resuming
    p.add_argument("--load_model",         type=str,   default=None)
    p.add_argument("--load_vecnorm",       type=str,   default=None)
    p.add_argument("--start_generation",   type=int,   default=1,
                   help="Generation index to start from (for resuming)")

    # Elo
    p.add_argument("--elo_k",             type=float, default=32.0)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SELF-PLAY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def self_play_train(args: argparse.Namespace, registry_snapshot: bytes | None = None) -> PPO:
    """
    Full AlphaZero-style self-play training loop.

    Generation lifecycle
    ────────────────────
    1. Sample opponent(s) from pool  (or use random for warmup gens)
    2. Rebuild SubprocVecEnv with fresh opponent assignments
    3. PPO.learn(timesteps_per_gen, reset_num_timesteps=False)
    4. Save checkpoint  gen_{N:03d}.zip + gen_{N:03d}_vecnorm.pkl
    5. Evaluate current model vs all pool entries → update Elo
    6. (Optional) evaluate vs random baseline
    7. Add current checkpoint to pool
    8. Print Elo leaderboard + save ratings

    Parameters
    ----------
    args : parsed argparse.Namespace from get_args()

    Returns
    -------
    The trained PPO model after all generations.
    """
    seed_everything(args.seed)
    _rng = random.Random(args.seed)   # seeded RNG for pool sampling

    # ── Output directories ────────────────────────────────────────────────────
    out_dir = os.path.join("models", args.run_name)
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    elo_path     = os.path.join(out_dir, "elo_ratings.json")
    eval_log_path = os.path.join(out_dir, "eval_log.csv")

    # ── State containers ──────────────────────────────────────────────────────
    pool     = OpponentPool(max_size=args.max_pool_size)
    elo      = EloRating.load(elo_path) if os.path.exists(elo_path) else EloRating(k_factor=args.elo_k)
    eval_log = EvalLog(eval_log_path)

    # ── Seed model: load checkpoint or create from scratch ───────────────────
    # Build an initial env for model construction (will be replaced each gen)
    init_env = build_train_env(
        n_envs             = args.n_envs,
        seed               = args.seed,
        opponent_mode      = "random",
        norm_obs           = not args.no_norm_obs,
        norm_reward        = not args.no_norm_reward,
        clip_obs           = args.clip_obs,
        gamma              = args.gamma,
        registry_snapshot  = registry_snapshot,
    )

    tb_log_dir = f"./logs/{args.run_name}"

    if args.load_model and os.path.exists(args.load_model + ".zip"):
        print(f"Resuming from {args.load_model}.zip  (generation {args.start_generation})")
        model = PPO.load(
            args.load_model,
            env            = init_env,
            tensorboard_log= tb_log_dir,
            seed           = args.seed,
        )
        if args.load_vecnorm and os.path.exists(args.load_vecnorm):
            tmp = VecNormalize.load(args.load_vecnorm, init_env.venv)
            init_env.obs_rms = tmp.obs_rms
            init_env.ret_rms = tmp.ret_rms
    else:
        print("Initialising new PPO model for self-play ...")
        model = PPO(
            policy          = "MlpPolicy",
            env             = init_env,
            verbose         = 1,
            tensorboard_log = tb_log_dir,
            learning_rate   = args.lr,
            n_steps         = args.n_steps,
            batch_size      = args.batch_size,
            n_epochs        = 10,
            ent_coef        = args.ent_coef,
            gamma           = args.gamma,
            gae_lambda      = args.gae_lambda,
            clip_range      = 0.2,
            device          = "auto",
            seed            = args.seed,
            policy_kwargs   = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    # ── Generation loop ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SELF-PLAY TRAINING  |  {args.generations} generations")
    print(f"  {args.timesteps_per_gen:,} timesteps / gen  |  {args.n_envs} parallel envs")
    print(f"  Pool size: {args.max_pool_size}  |  p_latest={args.p_latest:.0%}")
    print(f"{'═'*60}\n")

    current_train_env = init_env   # track the active training env for cleanup

    for gen in range(args.start_generation, args.start_generation + args.generations):
        in_warmup = gen <= args.start_generation + args.random_warmup_gens - 1

        print(f"\n{'─'*60}")
        print(f"  Generation {gen:03d} / {args.start_generation + args.generations - 1:03d}"
              f"  |  pool={pool!r}")
        print(f"  Phase: {'WARMUP (random)' if in_warmup else 'SELF-PLAY'}")
        print(f"{'─'*60}")

        # ── Build this generation's training env ──────────────────────────────
        if current_train_env is not init_env:
            current_train_env.close()   # close previous gen's subprocess pool

        if in_warmup or pool.size == 0:
            train_env = build_train_env(
                n_envs             = args.n_envs,
                seed               = args.seed + gen * 1000,
                opponent_mode      = "random",
                norm_obs           = not args.no_norm_obs,
                norm_reward        = not args.no_norm_reward,
                clip_obs           = args.clip_obs,
                gamma              = args.gamma,
                registry_snapshot  = registry_snapshot,
            )
        else:
            train_env = build_self_play_env(
                pool               = pool,
                n_envs             = args.n_envs,
                seed               = args.seed + gen * 1000,
                p_latest           = args.p_latest,
                norm_obs           = not args.no_norm_obs,
                norm_reward        = not args.no_norm_reward,
                clip_obs           = args.clip_obs,
                gamma              = args.gamma,
                rng                = _rng,
                registry_snapshot  = registry_snapshot,
            )

        # Transfer VecNorm stats from previous env into the new one so
        # the running mean/var is NOT reset between generations.
        if gen > args.start_generation:
            train_env.obs_rms = current_train_env.obs_rms
            train_env.ret_rms = current_train_env.ret_rms

        current_train_env = train_env
        model.set_env(train_env)

        # ── PPO learn for this generation ──────────────────────────────────────
        gen_cb = GenerationProgressCallback(generation=gen, verbose=0)
        model.learn(
            total_timesteps    = args.timesteps_per_gen,
            callback           = gen_cb,
            reset_num_timesteps= False,   # global timestep counter is cumulative
            progress_bar       = True,
        )

        # ── Save checkpoint ────────────────────────────────────────────────────
        ckpt_path  = os.path.join(out_dir, f"gen_{gen:03d}")
        vnorm_path = ckpt_path + "_vecnorm.pkl"

        model.save(ckpt_path)
        save_model_metadata(ckpt_path)
        train_env.save(vnorm_path)
        print(f"\n  Saved: {ckpt_path}.zip")

        # ── Evaluate vs random baseline ────────────────────────────────────────
        if args.eval_vs_random:
            rand_metrics = evaluate(
                model          = model,
                vecnorm_path   = vnorm_path,
                opponent_mode  = "random",
                episodes       = args.eval_episodes,
                seed           = args.seed + gen,
                deterministic  = True,
            )
            print_eval_results(rand_metrics, label="vs random baseline")

        # ── Evaluate vs pool + update Elo ──────────────────────────────────────
        if pool.size > 0:
            print(f"\n  Evaluating vs pool ({pool.size} entries, {args.eval_episodes} ep each):")
            evaluate_vs_pool(
                model        = model,
                pool         = pool,
                elo          = elo,
                eval_log     = eval_log,
                current_ver  = gen,
                generation   = gen,
                episodes     = args.eval_episodes,
                vecnorm_path = vnorm_path,
                seed         = args.seed + gen * 7,
            )
            elo.save(elo_path)
            print("\n  Elo leaderboard:")
            elo.print_leaderboard()

        # ── Add current checkpoint to pool ────────────────────────────────────
        pool.add(PoolEntry(
            version      = gen,
            model_path   = ckpt_path,
            vecnorm_path = vnorm_path,
        ))

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(out_dir, "final_model")
    model.save(final_path)
    save_model_metadata(final_path)
    current_train_env.save(final_path + "_vecnorm.pkl")
    elo.save(elo_path)

    print(f"\n{'═'*60}")
    print(f"  Self-play training complete!")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Elo ratings  : {elo_path}")
    print(f"  Eval log     : {eval_log_path}")
    print(f"{'═'*60}")

    elo.print_leaderboard()

    current_train_env.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SELF-PLAY EVALUATION TOOL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_versions(
    run_name:   str   = "self_play_v2",
    episodes:   int   = 100,
    seed:       int   = SEED,
) -> None:
    """
    Standalone eval: load every gen_*.zip in the run directory and evaluate
    each against a random baseline + its most recent predecessor.

    Prints a leaderboard and loads the existing Elo ratings if available.
    """
    out_dir  = os.path.join("models", run_name)
    elo_path = os.path.join(out_dir, "elo_ratings.json")

    if not os.path.isdir(out_dir):
        print(f"No run directory found at {out_dir}")
        return

    # Collect all gen_*.zip files
    snapshots = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("gen_") and name.endswith(".zip"):
            ver_str = name[4:-4]
            try:
                ver = int(ver_str)
            except ValueError:
                continue
            base        = os.path.join(out_dir, name[:-4])
            vnorm       = base + "_vecnorm.pkl"
            snapshots.append((ver, base, vnorm if os.path.exists(vnorm) else None))

    if not snapshots:
        print("No generation snapshots found.")
        return

    elo = EloRating.load(elo_path) if os.path.exists(elo_path) else EloRating()

    print(f"\nEvaluating {len(snapshots)} snapshots in {out_dir}\n")
    for ver, path, vnorm in snapshots:
        try:
            m = PPO.load(path)
        except Exception as e:
            print(f"  gen_{ver:03d}: FAILED to load ({e})")
            continue
        metrics = evaluate(m, vecnorm_path=vnorm, opponent_mode="random",
                           episodes=episodes, seed=seed)
        print(f"  gen_{ver:03d}  WR={metrics['win_rate']:.1%}"
              f"  R={metrics['avg_reward']:+.3f}"
              f"  Elo={elo.get_rating(ver):.0f}")

    print("\nFinal Elo leaderboard:")
    elo.print_leaderboard()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    # ── Step 1: Initialize registry — SINGLE ENTRYPOINT, STRICT ORDER ───────────
    from src.sprite_registry import (
        DEFAULT_INDEX_PATH,
        export_registry_snapshot,
        init_sprite_registry,
    )
    sprite_index = getattr(args, "sprite_index", None) or DEFAULT_INDEX_PATH
    sprite_meta  = init_sprite_registry(
        sprite_index,
        expected_version=getattr(args, "sprite_version", None),
    )
    registry_snapshot = export_registry_snapshot()
    print(
        f"\n[sprite_registry] Initialized\n"
        f"  version  : {sprite_meta.version}\n"
        f"  count    : {sprite_meta.count} slugs\n"
        f"  sha256   : {sprite_meta.sha256[:32]}…\n"
        f"  snapshot : {len(registry_snapshot):,} bytes\n"
    )

    self_play_train(args, registry_snapshot=registry_snapshot)


if __name__ == "__main__":
    main()
