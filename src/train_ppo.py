"""
train_ppo.py
~~~~~~~~~~~~
Full PPO training pipeline for the Pokémon RL system.

Upgrades over train_ia.py
─────────────────────────
• SubprocVecEnv  — N parallel workers (real multiprocessing, no GIL)
• VecNormalize   — running-mean normalisation for obs AND reward scaling
• WinRateCallback        — logs battle win-rate from Monitor episode infos
• VecNormCheckpointCallback — saves model + VecNormalize stats every K steps
• EvalCallback with VecNorm sync — eval env always uses training-time stats
• Opponent curriculum  — starts random, optional warm-start from prior model
• Full CLI args for every major hyperparameter

ARCHITECTURE
────────────
    ┌─────────────────────────────────────────────────────────┐
    │  SubprocVecEnv(N)  →  VecNormalize  →  PPO (MlpPolicy) │
    │       ↓ eval                                             │
    │  DummyVecEnv(1)    →  VecNormalize(training=False)      │
    │                         (stats synced from train VecNorm)│
    └─────────────────────────────────────────────────────────┘

USAGE
─────
    # Standard run — 8 envs, 500 k timesteps
    python -m src.train_ppo

    # Custom
    python -m src.train_ppo --n_envs 16 --total_timesteps 1_000_000

    # Continue from checkpoint
    python -m src.train_ppo \\
        --load_model   models/checkpoints/ppo_ckpt_200000_steps \\
        --load_vecnorm models/checkpoints/ppo_ckpt_200000_steps_vecnorm.pkl

    # Opponent curriculum: start against a warm opponent
    python -m src.train_ppo --opponent_model models/best_model_s3/best_model

OUTPUT FILES
────────────
    models/checkpoints/ppo_ckpt_{N}_steps.zip       — model checkpoint
    models/checkpoints/ppo_ckpt_{N}_steps_vecnorm.pkl  — VecNormalize stats
    models/best_ppo/best_model.zip                  — best eval checkpoint
    models/best_ppo/best_model_vecnorm.pkl          — corresponding VecNorm
    logs/ppo_*/                                     — TensorBoard logs
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

try:
    from src.env.pokemon_env import PokemonEnv
    from src.model_compat import save_model_metadata
    from src.reward_config import RewardExplainer
except ImportError:
    from env.pokemon_env import PokemonEnv
    from model_compat import save_model_metadata
    from reward_config import RewardExplainer


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
EVAL_FREQ     = 10_000   # steps between eval runs (per-env clock)
EVAL_EPISODES = 50       # episodes per eval run
CKPT_FREQ     = 50_000   # steps between model checkpoints


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    rank:              int,
    seed:              int          = SEED,
    opponent_mode:     str          = "random",
    opponent_model=None,
    registry_snapshot: bytes | None = None,
) -> callable:
    """
    Return a thunk (zero-arg callable) that creates a single monitored env.

    SubprocVecEnv requires a list of thunks — each spawned process calls its
    thunk to create an independent env.  Seeding by rank ensures every worker
    starts from a different RNG state while remaining reproducible.

    SPAWN WORKER INJECTION
    ──────────────────────
    registry_snapshot carries the pre-serialized sprite registry produced by
    export_registry_snapshot() in main().  The thunk calls
    load_registry_snapshot() before PokemonEnv() so the registry is available
    without any JSON parsing or file I/O inside the worker.

    Under fork: load_registry_snapshot() is a no-op (registry already
    inherited from parent process via copy-on-write).
    Under spawn: one pickle.loads() per worker at startup — never per step.

    PokemonEnv NEVER initializes the registry.  The env is a pure simulation.

    Parameters
    ----------
    rank              : worker index (0-based). Added to seed for diversity.
    seed              : base RNG seed.
    opponent_mode     : "random" | "greedy" | "model" | "mixed"
    opponent_model    : pre-loaded PPO model for "model" / "mixed" mode
    registry_snapshot : bytes from export_registry_snapshot(), or None
    """
    def _init() -> Monitor:
        # Inject sprite registry into this process — zero JSON parsing.
        # No-op under fork; fast pickle.loads() under spawn.
        if registry_snapshot is not None:
            from src.sprite_registry import load_registry_snapshot
            load_registry_snapshot(registry_snapshot)
        env = PokemonEnv()
        env.reset(seed=seed + rank)
        env.set_opponent(mode=opponent_mode, model=opponent_model)
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class WinRateCallback(BaseCallback):
    """
    Logs battle win-rate from Monitor episode info dicts.

    SB3's Monitor wrapper writes episode statistics into info["episode"]
    after each completed episode.  This callback reads the is_win flag
    set by PokemonEnv._build_turn_info() and computes a rolling win rate
    over the last win_window episodes across all parallel envs.

    Logged keys (visible in TensorBoard)
    ─────────────────────────────────────
    battle/win_rate_100    rolling win rate over last 100 episodes
    battle/episode_reward  last episode cumulative reward
    battle/episode_length  last episode turn count
    """

    def __init__(self, win_window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.win_window = win_window
        self._win_history: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Monitor wraps episode stats in info["episode"]
            ep = info.get("episode")
            if ep:
                self.logger.record("battle/episode_reward", float(ep["r"]))
                self.logger.record("battle/episode_length", float(ep["l"]))

            # PokemonEnv sets is_win=True in the final-step info
            if info.get("is_win") is not None and info.get("episode_reward") is not None:
                self._win_history.append(1.0 if info["is_win"] else 0.0)

            # Reward breakdown — log dominant components
            rb = info.get("reward_breakdown")
            if rb:
                self.logger.record("reward/damage",    float(rb.get("damage_reward", 0)))
                self.logger.record("reward/ko_net",    float(rb.get("ko_reward", 0)))
                self.logger.record("reward/matchup",   float(rb.get("matchup_shaping", 0)))
                self.logger.record("reward/threat",    float(rb.get("temporal_risk", 0)))
                self.logger.record("reward/momentum",  float(rb.get("momentum_reward", 0)))
                self.logger.record("reward/switch",    float(rb.get("smart_switch", 0)))
                self.logger.record("reward/move_qual", float(rb.get("move_quality", 0)))

        if self._win_history:
            recent = self._win_history[-self.win_window:]
            self.logger.record("battle/win_rate_100", float(np.mean(recent)))

        return True


class VecNormCheckpointCallback(BaseCallback):
    """
    Save a model checkpoint AND the matching VecNormalize statistics together.

    SB3's built-in CheckpointCallback saves the model but not VecNormalize.
    When you load a checkpoint later, the normalisation stats must match or
    the observation distribution seen by the model will be wrong.

    This callback saves both files atomically at every ckpt_freq steps.

    Output pattern
    ──────────────
    {save_path}/ppo_ckpt_{n_steps}_steps.zip
    {save_path}/ppo_ckpt_{n_steps}_steps_vecnorm.pkl
    """

    def __init__(
        self,
        save_path:  str,
        ckpt_freq:  int,
        name_prefix: str = "ppo_ckpt",
        verbose:     int = 0,
    ):
        super().__init__(verbose)
        self.save_path   = save_path
        self.ckpt_freq   = ckpt_freq
        self.name_prefix = name_prefix
        self._last_saved  = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved < self.ckpt_freq:
            return True

        self._last_saved = self.num_timesteps
        n = self.num_timesteps

        model_path  = os.path.join(self.save_path, f"{self.name_prefix}_{n}_steps")
        vnorm_path  = model_path + "_vecnorm.pkl"

        self.model.save(model_path)

        # Save VecNormalize stats if the training env is wrapped in one
        env = self.training_env
        if isinstance(env, VecNormalize):
            env.save(vnorm_path)
            if self.verbose:
                print(f"[VecNormCheckpoint] Saved {model_path}.zip + vecnorm stats")
        else:
            if self.verbose:
                print(f"[VecNormCheckpoint] Saved {model_path}.zip (no VecNormalize)")

        return True


class SyncVecNormEvalCallback(EvalCallback):
    """
    EvalCallback that synchronises VecNormalize statistics from the training
    env into the eval env before each evaluation run.

    Without this sync, the eval env's running mean/var diverges from the
    training env's stats over time, causing the evaluated model to see a
    different observation distribution than it was trained on.

    Parameters
    ----------
    train_env : VecNormalize  — the training VecNormalize wrapper
    eval_env  : VecNormalize  — the eval VecNormalize wrapper (training=False)
    All other kwargs forwarded to EvalCallback.
    """

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, **kwargs):
        super().__init__(eval_env=eval_env, **kwargs)
        self._train_env = train_env

    def _on_step(self) -> bool:
        # Sync stats into eval env before EvalCallback triggers its eval run
        if isinstance(self.eval_env, VecNormalize) and isinstance(self._train_env, VecNormalize):
            # Copy running statistics (obs mean/var, reward mean/var) without
            # changing the eval env's training=False flag
            self.eval_env.obs_rms    = self._train_env.obs_rms
            self.eval_env.ret_rms    = self._train_env.ret_rms
            self.eval_env.clip_obs   = self._train_env.clip_obs
            self.eval_env.clip_reward = self._train_env.clip_reward

        return super()._on_step()


# ─────────────────────────────────────────────────────────────────────────────
#  ENV BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_train_env(
    n_envs:            int,
    seed:              int          = SEED,
    opponent_mode:     str          = "random",
    opponent_model=None,
    norm_obs:          bool         = True,
    norm_reward:       bool         = True,
    clip_obs:          float        = 10.0,
    gamma:             float        = 0.99,
    registry_snapshot: bytes | None = None,
) -> VecNormalize:
    """
    Build the training VecEnv stack.

    Stack: SubprocVecEnv(N) → VecNormalize(norm_obs, norm_reward)

    Parameters
    ----------
    n_envs        : number of parallel workers
    seed          : base RNG seed (each worker offset by rank)
    opponent_mode : opponent policy for all workers
    opponent_model: PPO model for "model" / "mixed" opponent modes
    norm_obs      : normalise observations with running mean/std
    norm_reward   : scale rewards with running std (NOT mean — prevents
                    shifting the zero baseline)
    clip_obs      : clip normalised observations to [−clip_obs, +clip_obs]
    gamma         : discount factor (used by VecNormalize reward normalisation)

    Notes on VecNormalize
    ─────────────────────
    • norm_obs=True: each obs dimension is standardised to ~N(0,1).
      PokemonEnv obs are already bounded [0,1], so this mainly helps
      when different dimensions have very different variances.
    • norm_reward=True: reward is divided by the running std of the
      discounted return, keeping gradients at a reasonable scale.
      The MEAN is NOT subtracted — that would shift the terminal +1/−1
      signal toward zero and damage learning.
    • clip_obs=10: clips extreme values that appear rarely at episode start.
    """
    _start_method = "spawn" if sys.platform == "win32" else "fork"
    envs = SubprocVecEnv(
        [make_env(rank=i, seed=seed, opponent_mode=opponent_mode,
                  opponent_model=opponent_model,
                  registry_snapshot=registry_snapshot)
         for i in range(n_envs)],
        start_method=_start_method,
    )
    venv = VecNormalize(
        envs,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        gamma=gamma,
        training=True,
    )
    return venv


def build_eval_env(
    seed:              int          = SEED + 9999,
    norm_obs:          bool         = True,
    clip_obs:          float        = 10.0,
    gamma:             float        = 0.99,
    registry_snapshot: bytes | None = None,
) -> VecNormalize:
    """
    Build the eval VecEnv stack (single env, no reward normalisation).

    Stack: DummyVecEnv(1) → VecNormalize(norm_obs=True, norm_reward=False,
                                          training=False)

    The stats are synchronised from the training env before every eval run
    by SyncVecNormEvalCallback.  norm_reward=False here because we want to
    evaluate the *raw* episode reward (not the normalised version).
    """
    eval_env = DummyVecEnv([make_env(rank=0, seed=seed, registry_snapshot=registry_snapshot)])
    eval_venv = VecNormalize(
        eval_env,
        norm_obs=norm_obs,
        norm_reward=False,    # never normalise reward on the eval env
        clip_obs=clip_obs,
        gamma=gamma,
        training=False,       # stats are never updated from eval episodes
    )
    return eval_venv


# ─────────────────────────────────────────────────────────────────────────────
#  SEEDING
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ARGS
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO training pipeline for the Pokémon RL system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training scale
    p.add_argument("--total_timesteps", type=int,   default=500_000)
    p.add_argument("--n_envs",          type=int,   default=8,
                   help="Number of parallel SubprocVecEnv workers")
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--run_name",        type=str,   default="ppo_run",
                   help="Sub-directory name under logs/ for TensorBoard")

    # Opponent curriculum
    p.add_argument("--opponent_mode",   type=str,   default="random",
                   choices=["random", "greedy", "model", "mixed"])
    p.add_argument("--opponent_model",  type=str,   default=None,
                   help="Path to .zip model file for model/mixed opponent")

    # Checkpointing / loading
    p.add_argument("--load_model",      type=str,   default=None,
                   help="Path to .zip checkpoint to resume training from")
    p.add_argument("--load_vecnorm",    type=str,   default=None,
                   help="Path to matching VecNormalize .pkl stats file")
    p.add_argument("--ckpt_dir",        type=str,   default="models/checkpoints",
                   help="Directory for periodic model checkpoints")
    p.add_argument("--best_dir",        type=str,   default="models/best_ppo")

    # PPO hyperparameters
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--n_steps",         type=int,   default=2048,
                   help="Steps per env per PPO update (total = n_steps × n_envs)")
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--n_epochs",        type=int,   default=10)
    p.add_argument("--ent_coef",        type=float, default=0.02,
                   help="Entropy bonus — increase to maintain exploration")
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_range",      type=float, default=0.2)

    # VecNormalize
    p.add_argument("--no_norm_obs",     action="store_true",
                   help="Disable observation normalisation")
    p.add_argument("--no_norm_reward",  action="store_true",
                   help="Disable reward normalisation")
    p.add_argument("--clip_obs",        type=float, default=10.0)

    # Eval
    p.add_argument("--eval_freq",       type=int,   default=EVAL_FREQ)
    p.add_argument("--eval_episodes",   type=int,   default=EVAL_EPISODES)

    # Sprite registry
    p.add_argument(
        "--sprite_index",
        type=str,
        default=None,
        help=(
            "Path to sprite_index.json.  Defaults to DEFAULT_INDEX_PATH. "
            "Used to pin a training run to a specific sprite asset version."
        ),
    )
    p.add_argument(
        "--sprite_version",
        type=str,
        default=None,
        help=(
            "Expected sprite index version string.  If provided, training "
            "fails fast when the index file version doesn't match. "
            "Use to enforce reproducibility across machines."
        ),
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace, registry_snapshot: bytes | None = None) -> PPO:
    """
    Build envs, model, callbacks → train → save final model + VecNorm stats.

    registry_snapshot is the bytes produced by export_registry_snapshot() in
    main().  It is threaded into every env factory so spawn workers receive
    the pre-built registry with zero JSON parsing.

    Returns the trained PPO model.
    """
    seed_everything(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    # ── Optional: load warm opponent ──────────────────────────────────────────
    opponent_model = None
    if args.opponent_model and os.path.exists(args.opponent_model + ".zip"):
        print(f"Loading warm opponent from {args.opponent_model}.zip")
        opponent_model = PPO.load(args.opponent_model)

    # ── Build envs ────────────────────────────────────────────────────────────
    print(f"\nBuilding {args.n_envs} parallel training envs ...")
    train_env = build_train_env(
        n_envs             = args.n_envs,
        seed               = args.seed,
        opponent_mode      = args.opponent_mode,
        opponent_model     = opponent_model,
        norm_obs           = not args.no_norm_obs,
        norm_reward        = not args.no_norm_reward,
        clip_obs           = args.clip_obs,
        gamma              = args.gamma,
        registry_snapshot  = registry_snapshot,
    )
    eval_env = build_eval_env(
        seed               = args.seed + 9999,
        norm_obs           = not args.no_norm_obs,
        clip_obs           = args.clip_obs,
        gamma              = args.gamma,
        registry_snapshot  = registry_snapshot,
    )

    # ── Load VecNorm stats if resuming ────────────────────────────────────────
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        print(f"Loading VecNormalize stats from {args.load_vecnorm}")
        # Load stats into a temp wrapper then copy attributes
        tmp = VecNormalize.load(args.load_vecnorm, train_env.venv)
        train_env.obs_rms  = tmp.obs_rms
        train_env.ret_rms  = tmp.ret_rms

    # ── Build or load model ───────────────────────────────────────────────────
    tb_log_dir = f"./logs/{args.run_name}"

    if args.load_model and os.path.exists(args.load_model + ".zip"):
        print(f"\nResuming training from {args.load_model}.zip")
        model = PPO.load(
            args.load_model,
            env            = train_env,
            tensorboard_log= tb_log_dir,
            seed           = args.seed,
        )
    else:
        print("\nInitialising new PPO model ...")
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            verbose         = 1,
            tensorboard_log = tb_log_dir,
            learning_rate   = args.lr,
            n_steps         = args.n_steps,
            batch_size      = args.batch_size,
            n_epochs        = args.n_epochs,
            ent_coef        = args.ent_coef,
            gamma           = args.gamma,
            gae_lambda      = args.gae_lambda,
            clip_range      = args.clip_range,
            device          = "auto",
            seed            = args.seed,
            policy_kwargs   = dict(
                # Two hidden layers of 256 units — large enough for a 28-dim obs,
                # small enough to train quickly on CPU.
                net_arch = dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

    # ── Best-model save path ──────────────────────────────────────────────────
    best_model_path  = os.path.join(args.best_dir, "best_model")
    best_vnorm_path  = os.path.join(args.best_dir, "best_model_vecnorm.pkl")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    win_rate_cb  = WinRateCallback(verbose=0)

    checkpoint_cb = VecNormCheckpointCallback(
        save_path   = args.ckpt_dir,
        ckpt_freq   = CKPT_FREQ,
        name_prefix = "ppo_ckpt",
        verbose     = 1,
    )

    eval_cb = SyncVecNormEvalCallback(
        train_env       = train_env,
        eval_env        = eval_env,
        best_model_save_path = args.best_dir,
        log_path        = "./logs/",
        eval_freq       = args.eval_freq,
        n_eval_episodes = args.eval_episodes,
        deterministic   = True,
        verbose         = 1,
    )

    callbacks = CallbackList([win_rate_cb, checkpoint_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting PPO training")
    print(f"  Total timesteps  : {args.total_timesteps:,}")
    print(f"  Parallel envs    : {args.n_envs}")
    print(f"  Steps per update : {args.n_steps} × {args.n_envs} = {args.n_steps * args.n_envs:,}")
    print(f"  Opponent mode    : {args.opponent_mode}")
    print(f"  TensorBoard      : {tb_log_dir}\n")

    model.learn(
        total_timesteps    = args.total_timesteps,
        callback           = callbacks,
        progress_bar       = True,
        reset_num_timesteps= False if args.load_model else True,
    )

    # ── Save final model + VecNorm ────────────────────────────────────────────
    final_path = f"models/ppo_final_{args.run_name}"
    model.save(final_path)
    save_model_metadata(final_path)
    train_env.save(final_path + "_vecnorm.pkl")

    # Also save VecNorm alongside the EvalCallback's best model
    if os.path.exists(best_model_path + ".zip"):
        train_env.save(best_vnorm_path)

    print(f"\nTraining complete.")
    print(f"  Final model  : {final_path}.zip")
    print(f"  VecNorm stats: {final_path}_vecnorm.pkl")
    print(f"  Best model   : {best_model_path}.zip")

    train_env.close()
    eval_env.close()

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION HELPER  (standalone, also importable from self_play.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    vecnorm_path:    str | None = None,
    opponent_mode:   str        = "random",
    opponent_model              = None,
    episodes:        int        = 100,
    seed:            int        = SEED + 1,
    deterministic:   bool       = True,
) -> dict:
    """
    Evaluate a PPO model against a given opponent.

    Handles VecNormalize correctly — if a vecnorm_path is given, loads the
    statistics so the model receives properly normalised observations.

    Parameters
    ----------
    model          : trained PPO model (or path string)
    vecnorm_path   : path to matching VecNormalize .pkl file (optional)
    opponent_mode  : "random" | "greedy" | "model" | "mixed"
    opponent_model : PPO model for model/mixed opponents
    episodes       : number of evaluation episodes
    seed           : base seed for the eval env
    deterministic  : use deterministic policy (True for eval, False for sampling)

    Returns
    -------
    dict with keys:
        win_rate, avg_reward, avg_length, avg_damage_dealt,
        avg_damage_received, avg_ko_count, avg_stalled_turns
    """
    if isinstance(model, str):
        model = PPO.load(model)

    # Build a single eval env with optional VecNormalize
    raw_env  = DummyVecEnv([make_env(rank=0, seed=seed, opponent_mode=opponent_mode,
                                     opponent_model=opponent_model)])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_env

    wins, rewards, lengths, dealt, received, ko_counts, stalled = [], [], [], [], [], [], []

    for ep in range(episodes):
        # VecEnv.reset() returns obs only (not a tuple like gym 0.26+).
        # DummyVecEnv / VecNormalize both return a single ndarray.
        _reset_result = env.reset()
        if isinstance(_reset_result, tuple):
            obs = _reset_result[0]
        else:
            obs = _reset_result
        done = False
        truncated = False
        info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
                truncated = False

            # VecEnv returns arrays — extract scalar
            if hasattr(done, "__len__"):
                done      = bool(done[0])
                truncated = bool(truncated[0]) if hasattr(truncated, "__len__") else False
                info      = info[0] if isinstance(info, (list, tuple)) else info

        wins.append(1.0 if info.get("is_win") else 0.0)
        rewards.append(info.get("episode_reward", 0.0))
        lengths.append(info.get("episode_length", 0))
        dealt.append(info.get("damage_dealt", 0.0))
        received.append(info.get("damage_received", 0.0))
        ko_counts.append(info.get("ko_count", 0))
        stalled.append(info.get("stalled_turns", 0))

    env.close()

    return {
        "win_rate":            float(np.mean(wins)),
        "avg_reward":          float(np.mean(rewards)),
        "avg_length":          float(np.mean(lengths)),
        "avg_damage_dealt":    float(np.mean(dealt)),
        "avg_damage_received": float(np.mean(received)),
        "avg_ko_count":        float(np.mean(ko_counts)),
        "avg_stalled_turns":   float(np.mean(stalled)),
    }


def print_eval_results(metrics: dict, label: str = "") -> None:
    """Pretty-print an evaluate() result dict."""
    header = f"  Eval: {label}" if label else "  Evaluation Results"
    print(f"\n{header}")
    print(f"  {'Win rate':<28} {metrics['win_rate']:.1%}")
    print(f"  {'Avg reward':<28} {metrics['avg_reward']:.3f}")
    print(f"  {'Avg battle length':<28} {metrics['avg_length']:.1f} turns")
    print(f"  {'Avg KOs dealt':<28} {metrics['avg_ko_count']:.2f}")
    print(f"  {'Avg damage dealt':<28} {metrics['avg_damage_dealt']:.3f}")
    print(f"  {'Avg damage received':<28} {metrics['avg_damage_received']:.3f}")
    print(f"  {'Avg stalled turns':<28} {metrics['avg_stalled_turns']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    # ── Step 1: Initialize sprite registry — SINGLE ENTRYPOINT ──────────────────
    #
    # STRICT ORDERING: init → export → create envs. Never inside env / callback.
    #
    # fork  (Linux): workers inherit _STATE via copy-on-write — zero cost.
    # spawn (Windows): workers receive the snapshot via closure; load_registry_snapshot()
    #   does one pickle.loads() at startup — no JSON parsing, no file I/O per worker.
    from src.sprite_registry import (
        DEFAULT_INDEX_PATH,
        export_registry_snapshot,
        init_sprite_registry,
    )
    sprite_index = args.sprite_index or DEFAULT_INDEX_PATH
    sprite_meta  = init_sprite_registry(
        sprite_index,
        expected_version=args.sprite_version,
    )
    # Serialize once — threaded into every env factory via closure.
    # Workers reconstruct via pickle.loads(), never via JSON re-parsing.
    registry_snapshot = export_registry_snapshot()

    print(
        f"\n[sprite_registry] Initialized\n"
        f"  version      : {sprite_meta.version}\n"
        f"  count        : {sprite_meta.count} slugs\n"
        f"  sha256       : {sprite_meta.sha256[:32]}…\n"
        f"  path         : {sprite_meta.path}\n"
        f"  snapshot     : {len(registry_snapshot):,} bytes\n"
    )

    model = train(args, registry_snapshot=registry_snapshot)

    # Quick post-training eval
    print("\n=== Post-training evaluation (100 episodes vs random) ===")
    metrics = evaluate(model, opponent_mode="random", episodes=100)
    print_eval_results(metrics, label="vs random opponent")

    print("\n=== Post-training evaluation (100 episodes vs greedy) ===")
    metrics_greedy = evaluate(model, opponent_mode="greedy", episodes=100)
    print_eval_results(metrics_greedy, label="vs greedy opponent")


if __name__ == "__main__":
    main()
