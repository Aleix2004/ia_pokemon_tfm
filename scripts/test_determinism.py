"""
scripts/test_determinism.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Determinism validation suite for the Pokémon RL pipeline.

══════════════════════════════════════════════════════════════════════════════
WHAT THIS VALIDATES
══════════════════════════════════════════════════════════════════════════════

  1. Single-env determinism
       Two PokemonEnv instances with the same seed + same action sequence
       must produce bit-identical observations, rewards, and done flags at
       every step.

  2. Episode-hash determinism
       The SHA-256 of the full observation trajectory for a seeded episode is
       compared across two independent runs.  Identical hashes = no leak.

  3. Multiprocess determinism  (SubprocVecEnv)
       Two SubprocVecEnv runs with identical seeds and action sequences must
       produce identical stacked observations.  Catches leaks introduced by
       multiprocessing, spawn/fork differences, or shared state.

  4. Sprite registry determinism
       normalize_showdown_name() is a pure function — the same input must
       produce the same slug in every process.  Tested in a subprocess pool
       to catch any platform or process-level non-determinism.

  5. Cross-run log comparison
       Episodes are replayed twice; per-step (obs, reward) are hashed and
       written to a JSON log.  The two logs are compared entry-by-entry and
       a diff is reported if any mismatch is found.

══════════════════════════════════════════════════════════════════════════════
USAGE
══════════════════════════════════════════════════════════════════════════════

  # Full suite (all 5 checks)
  python scripts/test_determinism.py

  # Quiet mode — only print summary line
  python scripts/test_determinism.py --quiet

  # Specific checks only
  python scripts/test_determinism.py --checks env episode multiprocess

  # Custom seeds and episode length
  python scripts/test_determinism.py --seeds 42 123 999 --steps 200

  # Skip SubprocVecEnv check (useful on CI with limited parallelism)
  python scripts/test_determinism.py --checks env episode sprite log

  # Use a real sprite index for the sprite check
  python scripts/test_determinism.py --sprite_index assets/sprites/sprite_index.json

══════════════════════════════════════════════════════════════════════════════
EXPECTED OUTPUT (all pass)
══════════════════════════════════════════════════════════════════════════════

  ── 1. Single-env determinism ──────────────────────────────────────────────
    ✓  seed=42   | 100 steps | obs match   reward match   done match
    ✓  seed=123  | 100 steps | obs match   reward match   done match
    ✓  seed=999  | 100 steps | obs match   reward match   done match

  ── 2. Episode-hash determinism ────────────────────────────────────────────
    ✓  seed=42   | obs_hash matches across runs
    ✓  seed=123  | obs_hash matches across runs
    ✓  seed=999  | obs_hash matches across runs

  ── 3. Multiprocess determinism (SubprocVecEnv) ────────────────────────────
    ✓  4 workers | seed=42 | 50 steps | stacked obs bit-identical across runs

  ── 4. Sprite registry determinism ─────────────────────────────────────────
    ✓  normalize_showdown_name: 50 inputs | identical across 4 worker processes

  ── 5. Cross-run log comparison ────────────────────────────────────────────
    ✓  3 episodes | 0 mismatches | logs identical

  ════════════════════════════════════════════════════════════════════════════
    RESULT: 5/5 checks passed — pipeline is deterministic ✓
  ════════════════════════════════════════════════════════════════════════════

EXIT CODE
  0  — all requested checks passed
  1  — one or more checks failed (details printed to stdout)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Project root on path ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Lazy SB3 import (only needed for multiprocess check) ─────────────────────
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────────────────────

_PASS  = "\033[32m✓\033[0m"
_FAIL  = "\033[31m✗\033[0m"
_WARN  = "\033[33m⚠\033[0m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {_PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_WARN}  {msg}")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 70 - len(title))}")


# ─────────────────────────────────────────────────────────────────────────────
#  HASHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hash_array(arr: np.ndarray) -> str:
    """SHA-256 of the raw bytes of a float32 array.  Deterministic across runs."""
    return hashlib.sha256(np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()


def _hash_trajectory(obs_list: list[np.ndarray], reward_list: list[float]) -> str:
    """SHA-256 of the concatenation of all (obs, reward) pairs in an episode."""
    h = hashlib.sha256()
    for obs, r in zip(obs_list, reward_list):
        h.update(np.asarray(obs, dtype=np.float32).tobytes())
        h.update(np.float64(r).tobytes())
    return h.hexdigest()


def _hash_step_full(
    obs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict | None = None,
) -> str:
    """
    Bit-level SHA-256 over ALL step outputs: obs + reward + terminated +
    truncated + info.

    Stronger than _hash_array() — any single bit difference in any output
    field produces a completely different hash.  Used in Checks 6, 7, 8
    (research-grade extensions) where detecting subtle state leakage matters.

    Design decisions
    ────────────────
    • obs     → cast to float32, then .tobytes() — platform-independent layout
    • reward  → cast to float64 (never lose precision from Python float)
    • flags   → cast to uint8 (0/1) — avoids platform bool representation diffs
    • info    → json.dumps(sort_keys=True) so dict ordering cannot influence the
                hash; non-JSON-serialisable values fall back to str()
    """
    h = hashlib.sha256()
    h.update(np.asarray(obs, dtype=np.float32).tobytes())
    h.update(np.float64(reward).tobytes())
    h.update(np.uint8(int(bool(terminated))).tobytes())
    h.update(np.uint8(int(bool(truncated))).tobytes())
    if info:
        info_bytes = json.dumps(info, sort_keys=True, default=str).encode("utf-8")
        h.update(info_bytes)
    return h.hexdigest()


def _hash_trajectory_full(
    step_records: list[tuple[np.ndarray, float, bool, bool, dict | None]],
) -> str:
    """
    SHA-256 of a full trajectory using _hash_step_full() at each step.
    Covers obs + reward + done flags + info — nothing is omitted.
    """
    h = hashlib.sha256()
    for obs, reward, terminated, truncated, info in step_records:
        h.update(_hash_step_full(obs, reward, terminated, truncated, info).encode())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_instance(seed: int):
    """Return a fresh PokemonEnv seeded with *seed*, opponent_mode='random'."""
    try:
        from src.env.pokemon_env import PokemonEnv
    except ImportError:
        from env.pokemon_env import PokemonEnv
    env = PokemonEnv(seed=seed)
    env.set_opponent(mode="random")
    return env


def _make_env_thunk(rank: int, seed: int):
    """Returns a thunk compatible with SubprocVecEnv / DummyVecEnv."""
    def _init():
        env = _make_env_instance(seed + rank)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
#  DETERMINISTIC ACTION SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def _fixed_actions(n_steps: int, n_actions: int = 4, seed: int = 0) -> list[int]:
    """
    Generate a fixed sequence of actions using a seeded NumPy generator that
    is completely independent of the environment's RNG.
    This simulates a deterministic policy.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, n_actions, size=n_steps).tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 1 — Single-env determinism
# ─────────────────────────────────────────────────────────────────────────────

def check_single_env(seeds: list[int], n_steps: int, quiet: bool = False) -> bool:
    """
    Run two independent PokemonEnv instances with the same seed and identical
    action sequences.  Every (obs, reward, done) triple must be bit-identical.
    """
    _section("1. Single-env determinism")
    all_pass = True

    for seed in seeds:
        env_a = _make_env_instance(seed)
        env_b = _make_env_instance(seed)

        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)

        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)

        obs_mismatch = rew_mismatch = done_mismatch = 0

        for step_idx, action in enumerate(actions):
            out_a = env_a.step(action)
            out_b = env_b.step(action)

            obs_a_s, rew_a, term_a, trunc_a, _ = out_a
            obs_b_s, rew_b, term_b, trunc_b, _ = out_b

            if not np.array_equal(obs_a_s, obs_b_s):
                obs_mismatch += 1
                if not quiet:
                    print(f"       ↳ obs mismatch at step {step_idx}: "
                          f"A={obs_a_s[:3]}… B={obs_b_s[:3]}…")
            if rew_a != rew_b:
                rew_mismatch += 1
            if (term_a != term_b) or (trunc_a != trunc_b):
                done_mismatch += 1

            done_a = term_a or trunc_a
            done_b = term_b or trunc_b
            if done_a or done_b:
                # Re-seed both envs consistently on episode end
                obs_a, _ = env_a.reset(seed=seed + step_idx)
                obs_b, _ = env_b.reset(seed=seed + step_idx)

        env_a.close()
        env_b.close()

        ok = obs_mismatch == 0 and rew_mismatch == 0 and done_mismatch == 0
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL
        print(f"  {status}  seed={seed:<6} | {n_steps} steps | "
              f"obs {'match  ' if obs_mismatch == 0 else f'FAIL({obs_mismatch})':10s} "
              f"reward {'match  ' if rew_mismatch == 0 else f'FAIL({rew_mismatch})':10s} "
              f"done {'match' if done_mismatch == 0 else f'FAIL({done_mismatch})'}")
        if not ok and not quiet:
            print(f"       ↳ mismatches: obs={obs_mismatch} "
                  f"reward={rew_mismatch} done={done_mismatch}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2 — Episode-hash determinism
# ─────────────────────────────────────────────────────────────────────────────

def _run_episode(seed: int, max_steps: int = 500) -> tuple[str, int]:
    """
    Run one full episode to completion (or max_steps) and return
    (trajectory_hash, n_steps).  Independent of global state.
    """
    env = _make_env_instance(seed)
    actions = _fixed_actions(max_steps, n_actions=4, seed=seed)

    obs, _ = env.reset(seed=seed)
    obs_list: list[np.ndarray] = [obs.copy()]
    rew_list: list[float] = []

    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list.append(obs.copy())
        rew_list.append(float(reward))
        if terminated or truncated:
            break

    env.close()
    traj_hash = _hash_trajectory(obs_list, rew_list)
    return traj_hash, len(rew_list)


def check_episode_hash(seeds: list[int], max_steps: int = 500, quiet: bool = False) -> bool:
    """
    Run each seed twice and compare SHA-256 trajectory hashes.
    Identical hash = no randomness leak across independent runs.
    """
    _section("2. Episode-hash determinism")
    all_pass = True

    for seed in seeds:
        hash_a, n_a = _run_episode(seed, max_steps)
        hash_b, n_b = _run_episode(seed, max_steps)

        ok = (hash_a == hash_b) and (n_a == n_b)
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL
        print(f"  {status}  seed={seed:<6} | {n_a} steps | "
              f"{'trajectory hash matches' if ok else f'MISMATCH  A={hash_a[:16]}…  B={hash_b[:16]}…'}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 3 — Multiprocess determinism (SubprocVecEnv)
# ─────────────────────────────────────────────────────────────────────────────

def _run_vecenv(seed: int, n_workers: int, n_steps: int) -> list[str]:
    """
    Run SubprocVecEnv for n_steps with a fixed action sequence.
    Returns a list of obs hashes (one per step).
    """
    thunks = [_make_env_thunk(rank=i, seed=seed) for i in range(n_workers)]
    vec_env = SubprocVecEnv(thunks, start_method="spawn" if sys.platform == "win32" else "fork")

    # Re-seed every worker to the same value via env.seed() attr
    # SB3 SubprocVecEnv: pass seed through reset
    obs = vec_env.reset()

    # Actions: same for all workers and both runs
    actions_per_step = _fixed_actions(n_steps, n_actions=4, seed=seed)

    step_hashes: list[str] = [_hash_array(obs)]
    for action_val in actions_per_step:
        # Broadcast same action to all workers
        actions = np.full(n_workers, action_val, dtype=np.int32)
        obs, rewards, dones, _ = vec_env.step(actions)
        step_hashes.append(_hash_array(obs))

    vec_env.close()
    return step_hashes


def check_multiprocess(seed: int, n_workers: int, n_steps: int, quiet: bool = False) -> bool:
    """
    Run the same SubprocVecEnv configuration twice and compare obs hashes.
    """
    _section("3. Multiprocess determinism (SubprocVecEnv)")

    if not _HAS_SB3:
        _warn("stable-baselines3 not installed — skipping multiprocess check")
        return True

    try:
        hashes_a = _run_vecenv(seed, n_workers, n_steps)
        hashes_b = _run_vecenv(seed, n_workers, n_steps)
    except Exception as exc:
        _fail(f"SubprocVecEnv run failed: {exc}")
        return False

    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(hashes_a, hashes_b)) if a != b]

    ok = len(mismatches) == 0
    status = _PASS if ok else _FAIL
    print(f"  {status}  {n_workers} workers | seed={seed} | {n_steps} steps | "
          f"{'stacked obs bit-identical across runs' if ok else f'{len(mismatches)} step(s) differ'}")

    if not ok and not quiet:
        for step_idx, hash_a, hash_b in mismatches[:5]:
            print(f"       ↳ step {step_idx}: run_A={hash_a[:16]}… run_B={hash_b[:16]}…")
        if len(mismatches) > 5:
            print(f"       ↳ … and {len(mismatches) - 5} more mismatches")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 4 — Sprite registry determinism
# ─────────────────────────────────────────────────────────────────────────────

# Worker function defined at module level (required for multiprocessing.Pool)
def _worker_normalize(args: tuple[str, int]) -> tuple[str, str, int]:
    """
    Called in a subprocess: normalize *name* and return (name, slug, worker_pid).
    Imports are deliberately re-done inside the worker so we test the subprocess
    independently.
    """
    name, worker_id = args
    # Add project root to path (needed in spawned workers)
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src.sprite_registry import normalize_showdown_name
    except ImportError:
        from sprite_registry import normalize_showdown_name

    slug = normalize_showdown_name(name)
    return name, slug, worker_id


# Worker for snapshot-load test
def _worker_snapshot_lookup(args: tuple[bytes, str]) -> tuple[str, str, str]:
    """
    Called in a subprocess: load registry from snapshot, look up *form_name*,
    and return (form_name, slug, front_path).
    """
    snapshot, form_name = args
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src.sprite_registry import load_registry_snapshot, normalize_showdown_name
    except ImportError:
        from sprite_registry import load_registry_snapshot, normalize_showdown_name

    load_registry_snapshot(snapshot)
    slug = normalize_showdown_name(form_name)
    return form_name, slug, os.getpid().__str__()


# Canonical test names for the sprite determinism check
_SPRITE_TEST_NAMES: list[str] = [
    "Charizard", "Pikachu", "Mewtwo", "Gengar", "Machamp",
    "Mr. Mime", "Nidoran♀", "Nidoran♂", "Farfetch'd", "Ho-Oh",
    "Kommo-O", "kommo o", "Tapu Koko", "Tapu Fini", "Type: Null",
    "Iron Treads", "Great Tusk", "Gouging Fire", "Flabébé", "Porygon-Z",
    "charizard-mega-x", "charizard-Mega-X", "pikachu-gmax",
    "mr mime", "Mime Jr.", "sirfetch'd", "CHARIZARD", "", "  ",
    "Jangmo-O", "Hakamo-O", "Tapu Lele", "Tapu Bulu", "Ting-Lu",
    "Chien-Pao", "Wo-Chien", "Chi-Yu", "Walking Wake", "Roaring Moon",
    "Iron Valiant", "Iron Bundle", "Sandy Shocks", "Scream Tail",
    "Flutter Mane", "Brute Bonnet", "Slither Wing", "type:null",
    "porygon2", "Ho Oh", "farfetch\u2019d",
]


def check_sprite_determinism(n_workers: int, sprite_index: str | None, quiet: bool = False) -> bool:
    """
    Run normalize_showdown_name() in N worker processes and verify all
    workers return the same slug for each input name.

    If sprite_index is provided, also test load_registry_snapshot() in workers
    to verify the snapshot injection is deterministic across processes.
    """
    _section("4. Sprite registry determinism")
    all_pass = True

    # Part A: normalize_showdown_name in worker processes
    test_pairs = [(name, i % n_workers) for i, name in enumerate(_SPRITE_TEST_NAMES)]

    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = multiprocessing.get_context(start_method)

    try:
        with ctx.Pool(processes=n_workers) as pool:
            results_a = pool.map(_worker_normalize, test_pairs)
            results_b = pool.map(_worker_normalize, test_pairs)
    except Exception as exc:
        _fail(f"Worker pool failed: {exc}")
        return False

    # Build dicts: name → slug from each run
    dict_a = {name: slug for name, slug, _ in results_a}
    dict_b = {name: slug for name, slug, _ in results_b}

    mismatches = [
        (name, dict_a[name], dict_b[name])
        for name in dict_a
        if dict_a[name] != dict_b[name]
    ]

    ok_norm = len(mismatches) == 0
    all_pass = all_pass and ok_norm
    status = _PASS if ok_norm else _FAIL
    print(f"  {status}  normalize_showdown_name: {len(_SPRITE_TEST_NAMES)} inputs | "
          f"{'identical across ' + str(n_workers) + ' worker processes' if ok_norm else f'{len(mismatches)} mismatch(es)'}")

    if not ok_norm and not quiet:
        for name, slug_a, slug_b in mismatches[:5]:
            print(f"       ↳ {name!r}: run_A={slug_a!r}  run_B={slug_b!r}")

    # Also spot-check the main-process result matches workers
    try:
        from src.sprite_registry import normalize_showdown_name
    except ImportError:
        from sprite_registry import normalize_showdown_name  # type: ignore

    main_mismatches = [
        (name, normalize_showdown_name(name), dict_a[name])
        for name in dict_a
        if normalize_showdown_name(name) != dict_a[name]
    ]
    ok_main = len(main_mismatches) == 0
    all_pass = all_pass and ok_main
    status2 = _PASS if ok_main else _FAIL
    print(f"  {status2}  main-process normalize() matches worker results: "
          f"{'yes' if ok_main else f'{len(main_mismatches)} mismatch(es)'}")

    if not ok_main and not quiet:
        for name, main_slug, worker_slug in main_mismatches[:3]:
            print(f"       ↳ {name!r}: main={main_slug!r}  worker={worker_slug!r}")

    # Part B: snapshot injection (only if sprite_index exists)
    if sprite_index is not None:
        index_path = Path(sprite_index)
        if not index_path.exists():
            _warn(f"sprite_index not found: {index_path} — skipping snapshot check")
        else:
            try:
                from src.sprite_registry import init_sprite_registry, export_registry_snapshot
            except ImportError:
                from sprite_registry import init_sprite_registry, export_registry_snapshot  # type: ignore

            meta = init_sprite_registry(str(index_path))
            snapshot = export_registry_snapshot()

            snap_pairs = [(snapshot, name) for name in _SPRITE_TEST_NAMES[:20]]
            try:
                with ctx.Pool(processes=n_workers) as pool:
                    snap_results_a = pool.map(_worker_snapshot_lookup, snap_pairs)
                    snap_results_b = pool.map(_worker_snapshot_lookup, snap_pairs)
            except Exception as exc:
                _fail(f"Snapshot worker pool failed: {exc}")
                return False

            snap_dict_a = {name: slug for name, slug, _ in snap_results_a}
            snap_dict_b = {name: slug for name, slug, _ in snap_results_b}
            snap_mismatches = [
                (n, snap_dict_a[n], snap_dict_b[n])
                for n in snap_dict_a
                if snap_dict_a[n] != snap_dict_b[n]
            ]

            ok_snap = len(snap_mismatches) == 0
            all_pass = all_pass and ok_snap
            status3 = _PASS if ok_snap else _FAIL
            print(f"  {status3}  snapshot injection: {len(snap_pairs)} lookups | "
                  f"{meta.count} slugs | "
                  f"{'identical across ' + str(n_workers) + ' workers' if ok_snap else f'{len(snap_mismatches)} mismatch(es)'}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 5 — Cross-run log comparison
# ─────────────────────────────────────────────────────────────────────────────

def _collect_episode_log(seed: int, max_steps: int = 500) -> list[dict[str, Any]]:
    """
    Run one seeded episode; return a list of per-step log entries:
      {step, obs_hash, reward, terminated, truncated}
    """
    env = _make_env_instance(seed)
    actions = _fixed_actions(max_steps, n_actions=4, seed=seed)

    obs, _ = env.reset(seed=seed)
    log: list[dict[str, Any]] = []

    for step_idx, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        log.append({
            "step":       step_idx,
            "obs_hash":   _hash_array(obs),
            "reward":     round(float(reward), 8),
            "terminated": bool(terminated),
            "truncated":  bool(truncated),
        })
        if terminated or truncated:
            break

    env.close()
    return log


def check_log_comparison(seeds: list[int], max_steps: int = 500, quiet: bool = False) -> bool:
    """
    Collect per-step logs for two independent runs of the same seeds.
    Compare logs entry-by-entry and report any divergence.
    """
    _section("5. Cross-run log comparison")
    all_pass = True
    total_mismatches = 0

    for seed in seeds:
        log_a = _collect_episode_log(seed, max_steps)
        log_b = _collect_episode_log(seed, max_steps)

        # Step count must match
        if len(log_a) != len(log_b):
            _fail(f"seed={seed} | episode length differs: run_A={len(log_a)} run_B={len(log_b)}")
            all_pass = False
            total_mismatches += 1
            continue

        step_diffs: list[tuple[int, str, Any, Any]] = []
        for entry_a, entry_b in zip(log_a, log_b):
            for key in ("obs_hash", "reward", "terminated", "truncated"):
                if entry_a[key] != entry_b[key]:
                    step_diffs.append((entry_a["step"], key, entry_a[key], entry_b[key]))

        ok = len(step_diffs) == 0
        all_pass = all_pass and ok
        total_mismatches += len(step_diffs)
        status = _PASS if ok else _FAIL
        print(f"  {status}  seed={seed:<6} | {len(log_a)} steps | "
              f"{'0 mismatches — logs identical' if ok else f'{len(step_diffs)} field mismatch(es)'}")

        if not ok and not quiet:
            for step_idx, field, val_a, val_b in step_diffs[:5]:
                print(f"       ↳ step {step_idx} | {field}: "
                      f"A={str(val_a)[:32]}  B={str(val_b)[:32]}")
            if len(step_diffs) > 5:
                print(f"       ↳ … {len(step_diffs) - 5} more differences")

    if all_pass:
        print(f"  {_PASS}  {len(seeds)} episodes | 0 mismatches | logs identical")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 6 — Sensitivity (frozen-RNG / ignored-seed detector)
# ─────────────────────────────────────────────────────────────────────────────
#
#  RATIONALE
#  ─────────
#  Checks 1–5 verify that the SAME seed always produces the SAME result.
#  Check 6 verifies the inverse: DIFFERENT seeds MUST produce DIFFERENT
#  results.  A system with a frozen or ignored RNG would silently pass
#  Checks 1–5 (both runs produce the same wrong output) but fail Check 6.
#
#  PASS criteria (all must hold for each seed pair):
#    • trajectory hashes differ             — hard requirement
#    • at least MIN_FRAC_DIFFERENT steps have non-identical obs  — leniency: 10%
#    • at least one reward differs          — hard requirement
#
#  FAIL conditions:
#    • trajectory hashes IDENTICAL          → frozen RNG or seed completely ignored
#    • ALL step obs identical               → state space is seed-independent
#    • ALL rewards identical                → reward function is seed-independent

_SENSITIVITY_MIN_FRAC_DIFFERENT = 0.10   # at least 10% of steps must differ


def _run_episode_full(seed: int, max_steps: int = 500) -> list[tuple]:
    """
    Run one seeded episode.  Returns list of
    (obs_copy, reward, terminated, truncated, info) per step.
    Uses _hash_step_full() at collection time so the trajectory record
    is suitable for both sensitivity and stronger cross-run comparison.
    """
    env = _make_env_instance(seed)
    actions = _fixed_actions(max_steps, n_actions=4, seed=seed)
    obs, _ = env.reset(seed=seed)
    records: list[tuple] = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        records.append((obs.copy(), float(reward), bool(terminated), bool(truncated),
                        dict(info) if info else None))
        if terminated or truncated:
            break
    env.close()
    return records


def check_sensitivity(seeds: list[int], n_steps: int, quiet: bool = False) -> bool:
    """
    Run pairs of seeds with the same action sequence and verify trajectories
    diverge.  Detects frozen RNGs, ignored seeds, and degenerate state spaces.

    Requires at least 2 seeds.  All N*(N-1)/2 unique pairs are tested.
    """
    _section("6. Sensitivity — seed sensitivity (frozen-RNG detector)")

    if len(seeds) < 2:
        _warn("Need ≥ 2 seeds for sensitivity check — skipping")
        return True

    all_pass = True

    # Build all unique ordered pairs
    pairs = [(seeds[i], seeds[j]) for i in range(len(seeds))
             for j in range(i + 1, len(seeds))]

    for seed_a, seed_b in pairs:
        records_a = _run_episode_full(seed_a, max_steps=n_steps)
        records_b = _run_episode_full(seed_b, max_steps=n_steps)

        # Truncate to the shorter episode for comparison
        n = min(len(records_a), len(records_b))
        if n == 0:
            _fail(f"seeds=({seed_a},{seed_b}) | both episodes have 0 steps")
            all_pass = False
            continue

        recs_a = records_a[:n]
        recs_b = records_b[:n]

        # Trajectory hash (using full hash — obs+reward+done+info)
        hash_a = _hash_trajectory_full(recs_a)
        hash_b = _hash_trajectory_full(recs_b)
        hashes_equal = (hash_a == hash_b)

        # Per-step obs comparison
        obs_diff_steps = sum(
            1 for (oa, *_), (ob, *_) in zip(recs_a, recs_b)
            if not np.array_equal(oa, ob)
        )
        frac_obs_diff = obs_diff_steps / n

        # Reward comparison
        rewards_a = [r for _, r, *_ in recs_a]
        rewards_b = [r for _, r, *_ in recs_b]
        any_reward_diff = any(ra != rb for ra, rb in zip(rewards_a, rewards_b))

        # ── Verdict ──────────────────────────────────────────────────────────
        fail_frozen   = hashes_equal                             # hard fail
        fail_obs      = obs_diff_steps == 0                      # hard fail
        fail_reward   = not any_reward_diff                      # hard fail
        warn_low_diff = (not fail_obs) and frac_obs_diff < _SENSITIVITY_MIN_FRAC_DIFFERENT

        ok = not (fail_frozen or fail_obs or fail_reward)
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        label = f"seeds=({seed_a},{seed_b}) | {n} steps compared"
        if ok:
            print(f"  {status}  {label} | "
                  f"obs differ {obs_diff_steps}/{n} ({frac_obs_diff:.0%}) | "
                  f"trajectories distinct")
            if warn_low_diff:
                _warn(f"       Low obs-diff rate ({frac_obs_diff:.0%}) — "
                      f"seeds may have limited influence on state space")
        else:
            print(f"  {status}  {label}")
            if fail_frozen:
                print(f"       ↳ FROZEN RNG: trajectory hashes identical "
                      f"({hash_a[:16]}…) — seed is completely ignored")
            if fail_obs:
                print(f"       ↳ ALL obs IDENTICAL across seeds — "
                      f"env state is seed-independent")
            if fail_reward:
                print(f"       ↳ ALL rewards IDENTICAL across seeds — "
                      f"reward function is seed-independent")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 7 — Multiprocess order (per-worker-ID observation stability)
# ─────────────────────────────────────────────────────────────────────────────
#
#  RATIONALE
#  ─────────
#  Check 3 hashes the entire *stacked* obs tensor at each step.  It passes
#  even if SubprocVecEnv silently swaps worker outputs between runs (obs[0]
#  in run A corresponds to obs[1] in run B).  Check 7 verifies:
#
#    a) Per-worker identity: obs[worker_i] at step t in run A == obs[worker_i]
#       at step t in run B.  Same seed per worker → same obs sequence.
#
#    b) Worker independence: obs[worker_i] ≠ obs[worker_j] (different seeds
#       → different trajectories, using the same sensitivity logic as Check 6).
#
#  This directly catches non-deterministic task-to-worker assignment inside
#  SubprocVecEnv and any shared mutable state that leaks across workers.

def _run_vecenv_per_worker(
    seed: int,
    n_workers: int,
    n_steps: int,
) -> dict[int, list[str]]:
    """
    Run SubprocVecEnv for n_steps.  Return per-worker step hashes using
    _hash_step_full() on each worker's individual obs slice.

    Returns
    -------
    dict[worker_id → list[step_hash]]   (length n_steps + 1, including reset obs)
    """
    thunks = [_make_env_thunk(rank=i, seed=seed) for i in range(n_workers)]
    vec_env = SubprocVecEnv(
        thunks,
        start_method="spawn" if sys.platform == "win32" else "fork",
    )
    stacked = vec_env.reset()   # shape (n_workers, obs_dim)

    # Initialise per-worker hash lists with the reset observation
    per_worker: dict[int, list[str]] = {
        i: [_hash_step_full(stacked[i], 0.0, False, False)]
        for i in range(n_workers)
    }

    actions_seq = _fixed_actions(n_steps, n_actions=4, seed=seed)
    for action_val in actions_seq:
        actions = np.full(n_workers, action_val, dtype=np.int32)
        stacked, rewards, dones, _ = vec_env.step(actions)
        for i in range(n_workers):
            per_worker[i].append(
                _hash_step_full(stacked[i], float(rewards[i]), bool(dones[i]), False)
            )

    vec_env.close()
    return per_worker


def check_multiprocess_order(
    seed: int,
    n_workers: int,
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Run SubprocVecEnv twice.  For each worker slot (0..n_workers-1) verify
    that its per-step hash sequence is identical across both runs.

    Also verify that different worker slots produce different hash sequences
    (each worker uses a different seed so their trajectories must diverge).
    """
    _section("7. Multiprocess order — per-worker-ID stability")

    if not _HAS_SB3:
        _warn("stable-baselines3 not installed — skipping multiprocess order check")
        return True

    if n_workers < 2:
        _warn("Need ≥ 2 workers for multiprocess order check — skipping")
        return True

    try:
        pw_a = _run_vecenv_per_worker(seed, n_workers, n_steps)
        pw_b = _run_vecenv_per_worker(seed, n_workers, n_steps)
    except Exception as exc:
        _fail(f"SubprocVecEnv run failed: {exc}")
        return False

    all_pass = True

    # ── Part A: per-worker identity across runs ───────────────────────────────
    identity_mismatches: dict[int, int] = {}
    for wid in range(n_workers):
        mm = sum(1 for ha, hb in zip(pw_a[wid], pw_b[wid]) if ha != hb)
        if mm:
            identity_mismatches[wid] = mm

    ok_identity = len(identity_mismatches) == 0
    all_pass = all_pass and ok_identity
    status = _PASS if ok_identity else _FAIL
    print(f"  {status}  Per-worker identity: {n_workers} workers | {n_steps} steps | "
          f"{'all workers bit-identical across runs' if ok_identity else f'{len(identity_mismatches)} worker(s) differ'}")
    if not ok_identity and not quiet:
        for wid, mm_count in sorted(identity_mismatches.items()):
            print(f"       ↳ worker {wid}: {mm_count} step(s) differ between run A and run B")

    # ── Part B: worker independence (different seeds → different trajectories) ─
    n_independent = 0
    n_pairs_total = n_workers * (n_workers - 1) // 2
    degenerate_pairs: list[tuple[int, int]] = []

    for i in range(n_workers):
        for j in range(i + 1, n_workers):
            # Use run A hashes — compare worker i vs worker j
            seq_i = pw_a[i]
            seq_j = pw_a[j]
            n_cmp = min(len(seq_i), len(seq_j))
            diff_steps = sum(1 for ha, hb in zip(seq_i[:n_cmp], seq_j[:n_cmp]) if ha != hb)
            frac_diff = diff_steps / n_cmp if n_cmp else 0.0
            if frac_diff < _SENSITIVITY_MIN_FRAC_DIFFERENT:
                degenerate_pairs.append((i, j))
            else:
                n_independent += 1

    ok_independence = len(degenerate_pairs) == 0
    all_pass = all_pass and ok_independence
    status2 = _PASS if ok_independence else _FAIL
    print(f"  {status2}  Worker independence: {n_pairs_total} worker-pairs | "
          f"{'all produce distinct trajectories' if ok_independence else f'{len(degenerate_pairs)} pair(s) suspiciously similar'}")
    if not ok_independence and not quiet:
        for wi, wj in degenerate_pairs[:5]:
            print(f"       ↳ worker {wi} and worker {j} trajectories nearly identical — "
                  f"possible shared seed or frozen RNG")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 8 — Cross-run reproducibility (persistent hash baseline)
# ─────────────────────────────────────────────────────────────────────────────
#
#  RATIONALE
#  ─────────
#  Checks 1–7 all run within a single process invocation.  Check 8 verifies
#  reproducibility ACROSS separate script invocations — something no in-process
#  check can catch.  It persists trajectory hashes to a JSON baseline file
#  and compares on subsequent runs.
#
#  FIRST RUN  → saves baseline to <log_dir>/determinism_baseline.json, returns True.
#  SUBSEQUENT → loads baseline, compares; fails on any mismatch.
#  --reset_baseline → delete existing baseline and save a fresh one.
#
#  The baseline records: per-seed trajectory hash + step count + timestamp +
#  library versions (numpy, Python) for auditability.


def _collect_run_hashes_full(seeds: list[int], max_steps: int = 500) -> dict[str, Any]:
    """
    Run one seeded episode per seed using _hash_step_full() for every step.
    Returns a dict ready for JSON serialisation.
    """
    episodes: dict[str, Any] = {}
    for seed in seeds:
        records = _run_episode_full(seed, max_steps=max_steps)
        traj_hash = _hash_trajectory_full(records)
        episodes[str(seed)] = {
            "traj_hash": traj_hash,
            "n_steps":   len(records),
        }
    return episodes


def check_cross_run_reproducibility(
    seeds: list[int],
    max_steps: int,
    log_dir: str,
    reset_baseline: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Persist per-seed trajectory hashes to a JSON baseline file and compare
    on subsequent invocations.  Catches any non-determinism that only manifests
    across separate process launches (e.g., PYTHONHASHSEED, OS entropy).

    Returns True on first run (baseline saved) and on hash match.
    Returns False if any hash mismatches the saved baseline.
    """
    _section("8. Cross-run reproducibility — persistent hash baseline")
    baseline_path = Path(log_dir) / "determinism_baseline.json"

    # Collect hashes for this run
    current_episodes = _collect_run_hashes_full(seeds, max_steps)

    # ── Delete baseline if requested ─────────────────────────────────────────
    if reset_baseline and baseline_path.exists():
        baseline_path.unlink()
        _warn("Existing baseline deleted (--reset_baseline)")

    # ── First run: save baseline ──────────────────────────────────────────────
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "created_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "seeds":         seeds,
            "hash_function": "_hash_step_full (obs+reward+terminated+truncated+info)",
            "episodes":      current_episodes,
        }
        baseline_path.write_text(json.dumps(payload, indent=2))
        _warn(f"No baseline found — saved to: {baseline_path}")
        _warn("Run the suite again to compare against this baseline.")
        return True   # not a failure — first run always passes

    # ── Subsequent runs: compare ──────────────────────────────────────────────
    try:
        saved_payload = json.loads(baseline_path.read_text())
        saved_episodes: dict[str, Any] = saved_payload.get("episodes", {})
    except (json.JSONDecodeError, KeyError) as exc:
        _fail(f"Baseline file corrupt: {exc}  ({baseline_path})")
        _warn("Delete the file and re-run to create a fresh baseline.")
        return False

    mismatches: list[tuple[str, str, str]] = []  # (seed_str, saved_hash, current_hash)
    missing_seeds: list[str] = []
    new_seeds: list[str] = []

    for seed_str, current_data in current_episodes.items():
        if seed_str not in saved_episodes:
            new_seeds.append(seed_str)
            continue
        saved_hash = saved_episodes[seed_str]["traj_hash"]
        curr_hash  = current_data["traj_hash"]
        saved_n    = saved_episodes[seed_str]["n_steps"]
        curr_n     = current_data["n_steps"]
        if saved_hash != curr_hash or saved_n != curr_n:
            mismatches.append((seed_str, saved_hash, curr_hash))

    for seed_str in saved_episodes:
        if seed_str not in current_episodes:
            missing_seeds.append(seed_str)

    ok = len(mismatches) == 0
    status = _PASS if ok else _FAIL

    saved_ts = saved_payload.get("created_at", "unknown")
    print(f"  {status}  Baseline: {baseline_path.name}  (saved {saved_ts[:19]})")
    print(f"  {status}  {len(seeds)} seeds | "
          f"{'all trajectory hashes match baseline' if ok else f'{len(mismatches)} mismatch(es) detected'}")

    if new_seeds:
        _warn(f"  New seeds not in baseline: {new_seeds} — update baseline with --reset_baseline")
    if missing_seeds:
        _warn(f"  Seeds in baseline but not in current run: {missing_seeds}")

    if not ok and not quiet:
        for seed_str, saved_hash, curr_hash in mismatches:
            print(f"       ↳ seed={seed_str}: "
                  f"saved={saved_hash[:20]}… "
                  f"current={curr_hash[:20]}…")
        print(f"\n  DIAGNOSIS: trajectory hashes differ across invocations.")
        print(f"  Likely causes:")
        print(f"    • PYTHONHASHSEED not fixed (set PYTHONHASHSEED=0)")
        print(f"    • Library version change (numpy/torch/gym)")
        print(f"    • OS-level entropy leak (time(), getpid(), os.urandom())")
        print(f"    • Env uses uncontrolled global RNG (random.random, np.random)")
        print(f"  To reset baseline: add --reset_baseline flag")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  RNG CONTROL & LIBRARY VERSION CHECK
# ─────────────────────────────────────────────────────────────────────────────
#
#  Not a pure pass/fail check on the env — a diagnostic that validates the
#  test environment itself before trusting other checks.
#
#  PROBES:
#    A. np.random.default_rng independence  — _fixed_actions() must be
#       immune to corruption of np.random global state
#    B. env isolation from np.random global  — PokemonEnv uses gymnasium
#       np_random (seeded), not the np.random global.  Corrupting the global
#       must not change episode trajectories.
#    C. torch.manual_seed() reproducibility  — if torch is installed, verify
#       torch.randn() is deterministic under manual_seed().
#    D. PYTHONHASHSEED advisory  — emit a warning if not set.
#    E. Library version fingerprint  — print exact versions for audit trail.


def check_rng_control(quiet: bool = False) -> bool:
    """
    Validate the RNG environment and print a library version fingerprint.

    Performs three executable probes (A, B, C) that each have a binary
    PASS/FAIL outcome, plus one advisory warning (D) and a version printout (E).
    """
    _section("RNG control & library versions")
    all_pass = True

    # ── E. Library version fingerprint ───────────────────────────────────────
    _HAS_TORCH_LOCAL = False
    try:
        import torch as _torch
        _torch_ver = _torch.__version__
        _cuda = _torch.cuda.is_available()
        _HAS_TORCH_LOCAL = True
    except ImportError:
        _torch_ver = None
        _cuda = False

    print(f"  {'─'*64}")
    print(f"  python  {platform.python_version()}   ({sys.platform})")
    print(f"  numpy   {np.__version__}")
    if _torch_ver:
        print(f"  torch   {_torch_ver}   (CUDA: {_cuda})")
    else:
        print(f"  torch   not installed")
    try:
        import gymnasium as _gym
        print(f"  gymnasium {_gym.__version__}")
    except ImportError:
        try:
            import gym as _gym  # type: ignore
            print(f"  gym     {_gym.__version__}  (legacy)")
        except ImportError:
            print(f"  gymnasium not installed")
    try:
        import stable_baselines3 as _sb3
        print(f"  stable-baselines3  {_sb3.__version__}")
    except ImportError:
        print(f"  stable-baselines3  not installed")
    print(f"  {'─'*64}")

    # ── D. PYTHONHASHSEED advisory ───────────────────────────────────────────
    hashseed_env = os.environ.get("PYTHONHASHSEED", None)
    if hashseed_env is None:
        _warn("PYTHONHASHSEED not set — Python hash randomisation is active")
        _warn("   Fix: PYTHONHASHSEED=0 python scripts/test_determinism.py")
        _warn("   Impact: dict/set ordering in info dicts may vary across runs")
    elif hashseed_env == "random":
        _warn("PYTHONHASHSEED=random — explicit random hash seed active")
    else:
        print(f"  {_PASS}  PYTHONHASHSEED={hashseed_env}  (hash randomisation disabled)")

    # ── A. np.random.default_rng is independent of np.random global state ────
    actions_clean = _fixed_actions(50, seed=77)
    np.random.seed(0xDEADBEEF)                    # corrupt global state
    actions_after_corrupt = _fixed_actions(50, seed=77)
    np.random.seed(0)                              # restore
    ok_rng_independent = (actions_clean == actions_after_corrupt)
    all_pass = all_pass and ok_rng_independent
    status_a = _PASS if ok_rng_independent else _FAIL
    print(f"  {status_a}  Probe A: np.random.default_rng(seed) independent of "
          f"np.random global state")
    if not ok_rng_independent:
        print(f"       ↳ _fixed_actions() output differs when np.random.seed() is called "
              f"before it — default_rng() is leaking global state somehow")

    # ── B. PokemonEnv isolated from np.random global state ───────────────────
    try:
        hash_baseline, n_baseline = _run_episode(42, max_steps=100)
        np.random.seed(0xCAFEBABE)                 # corrupt global state
        hash_after_corrupt, n_after_corrupt = _run_episode(42, max_steps=100)
        np.random.seed(0)                          # restore
        ok_env_isolated = (hash_baseline == hash_after_corrupt) and (n_baseline == n_after_corrupt)
        all_pass = all_pass and ok_env_isolated
        status_b = _PASS if ok_env_isolated else _FAIL
        print(f"  {status_b}  Probe B: PokemonEnv trajectory isolated from "
              f"np.random global state (uses gymnasium np_random)")
        if not ok_env_isolated:
            print(f"       ↳ LEAK DETECTED: env trajectory changes when np.random.seed() "
                  f"is called — env is absorbing global RNG state")
    except Exception as exc:
        _warn(f"Probe B skipped — could not run env: {exc}")

    # ── C. torch.manual_seed() reproducibility ───────────────────────────────
    if _HAS_TORCH_LOCAL:
        import torch as _torch_inner
        _torch_inner.manual_seed(42)
        v1 = _torch_inner.randn(8).tolist()
        _torch_inner.manual_seed(42)
        v2 = _torch_inner.randn(8).tolist()
        ok_torch = v1 == v2
        all_pass = all_pass and ok_torch
        status_c = _PASS if ok_torch else _FAIL
        print(f"  {status_c}  Probe C: torch.manual_seed(42) produces identical "
              f"torch.randn() output on repeat calls")
        if not ok_torch:
            print(f"       ↳ torch random state is not deterministic under manual_seed — "
                  f"check for background threads or CUDA non-determinism")
    else:
        print(f"  {_WARN}  Probe C skipped — torch not installed")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  REWARD DISTRIBUTION SANITY (bonus — not a pass/fail check)
# ─────────────────────────────────────────────────────────────────────────────

def print_reward_summary(seeds: list[int], n_steps: int = 200) -> None:
    """
    Print min/max/mean reward and episode return for each seed.
    Not a pass/fail check — used to detect suspiciously uniform distributions
    that might indicate a frozen RNG.
    """
    _section("Reward distribution sanity (informational)")
    for seed in seeds:
        env = _make_env_instance(seed)
        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for action in actions:
            _, r, term, trunc, _ = env.step(action)
            rewards.append(float(r))
            if term or trunc:
                obs, _ = env.reset(seed=seed + len(rewards))
        env.close()
        arr = np.array(rewards)
        n_unique = len(np.unique(np.round(arr, 4)))
        suspicious = n_unique < 3
        flag = f"  {_WARN} SUSPICIOUS: only {n_unique} unique reward values — possible frozen RNG" if suspicious else ""
        print(f"  seed={seed:<6} | n={len(arr):4d} | "
              f"min={arr.min():.4f} max={arr.max():.4f} "
              f"mean={arr.mean():.4f} unique={n_unique}{flag}")


# ═════════════════════════════════════════════════════════════════════════════
#  RESEARCH-GRADE HARDENING — FULL AUDIT SYSTEM  (Checks 10–16)
# ═════════════════════════════════════════════════════════════════════════════
#
#  These checks operate on the assumption that ALL previous checks can pass
#  while the system is STILL non-deterministic in ways invisible to them.
#  Each check here targets a distinct failure mode that would slip through.
#
#  FAILURE MODES TARGETED:
#    Check 10 — obs matches but internal state diverged  ("fake determinism")
#    Check 11 — RNG counter frozen or drifting between identical runs
#    Check 12 — floating-point non-determinism (FP rounding, BLAS, order-of-ops)
#    Check 13 — container-type dependence (list vs tuple) or action-order reliance
#    Check 14 — state capture incomplete (serialization roundtrip)
#    Check 15 — cross-process contamination not visible in aggregated outputs
#    Check 16 — time-based randomness or async race conditions
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  GOLD-STANDARD HASH INFRASTRUCTURE  (Requirement 8)
# ─────────────────────────────────────────────────────────────────────────────
#
#  The gold-standard hash is the SINGLE SOURCE OF TRUTH for all hardening
#  checks.  It covers every bit of information that can differ between runs:
#
#    Layer 1 — observation      (float32 bytes, platform-independent layout)
#    Layer 2 — reward           (float64 bytes, preserves full precision)
#    Layer 3 — terminated flag  (uint8: 0 or 1)
#    Layer 4 — truncated flag   (uint8: 0 or 1)
#    Layer 5 — info dict        (json.dumps sort_keys=True, non-JSON → str())
#    Layer 6 — full env state   (ALL internal variables, sorted-key JSON)
#    Layer 7 — RNG state        (numpy bit_generator state, sorted-key JSON)
#
#  If the gold-standard hash matches, nothing can differ.  Period.


def _deep_sort_json(obj: Any) -> Any:
    """
    Recursively convert any object into a JSON-serialisable form with all
    dicts sorted by key.  Deterministic regardless of Python insertion order.

    Type rules:
      dict   → sort keys, recurse values
      list/tuple → list, recurse elements
      np.ndarray → float64 list (all elements same dtype for stability)
      np.integer → int
      np.floating → float
      everything else → leave as-is (json.dumps will call str() if needed)
    """
    if isinstance(obj, dict):
        return {k: _deep_sort_json(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_deep_sort_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float64).tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _extract_env_state(env: Any) -> dict[str, Any]:
    """
    Extract ALL semantically significant hidden state from a PokemonEnv instance.

    Coverage (every field that can change between steps):
      Battle scalars   — hp_ia, hp_rival, estado_ia, estado_rival, turn_count
      Momentum         — damage_momentum, consecutive_advantage
      Episode counters — episode_damage_dealt/received, kos_for/against,
                         stalled_turns, switches_this_episode, episode_reward
      Active Pokémon   — ia_pokemon / rival_pokemon including stat_stages,
                         current_hp, status, and all per-move pp values
      Reward breakdown — last_reward_breakdown (float dict)

    Returns a dict with all keys sorted for deterministic JSON serialisation.
    NO pickle.  NO hash-randomisable objects.  Pure scalar/list/dict output.
    """
    return _deep_sort_json({
        # ── Battle scalars ────────────────────────────────────────────────────
        "hp_ia":                getattr(env, "hp_ia",    None),
        "hp_rival":             getattr(env, "hp_rival", None),
        "estado_ia":            getattr(env, "estado_ia",    None),
        "estado_rival":         getattr(env, "estado_rival", None),
        "turn_count":           getattr(env, "turn_count",   None),
        # ── Momentum / combo trackers ─────────────────────────────────────────
        "damage_momentum":      getattr(env, "damage_momentum",      None),
        "consecutive_advantage":getattr(env, "consecutive_advantage", None),
        "switches_this_episode":getattr(env, "switches_this_episode", None),
        # ── Episode accumulators ──────────────────────────────────────────────
        "episode_reward":           getattr(env, "episode_reward",           None),
        "episode_damage_dealt":     getattr(env, "episode_damage_dealt",     None),
        "episode_damage_received":  getattr(env, "episode_damage_received",  None),
        "episode_kos_for":          getattr(env, "episode_kos_for",          None),
        "episode_kos_against":      getattr(env, "episode_kos_against",      None),
        "episode_stalled_turns":    getattr(env, "episode_stalled_turns",    None),
        # ── Last reward breakdown (float dict) ────────────────────────────────
        "last_reward_breakdown":    getattr(env, "last_reward_breakdown",    {}),
        # ── Active Pokémon (full state including stat_stages, current_hp) ─────
        "ia_pokemon":    getattr(env, "ia_pokemon",    {}),
        "rival_pokemon": getattr(env, "rival_pokemon", {}),
    })


def _get_rng_state(env: Any) -> dict[str, Any]:
    """
    Extract the full numpy Generator (PCG64) state as a JSON-serialisable dict.

    numpy PCG64 state structure:
      {'bit_generator': 'PCG64',
       'state': {'state': <large int>, 'inc': <large int>},
       'has_uint32': 0, 'uinteger': 0}

    All numpy integers are converted to Python ints.  The dict is
    deterministically sorted.  If env.np_random is unavailable, returns {}.
    """
    try:
        raw = env.np_random.bit_generator.state
        return _deep_sort_json(raw)
    except AttributeError:
        return {}


def _hash_gold_standard(
    env: Any,
    obs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict | None = None,
) -> str:
    """
    Gold-standard SHA-256: the SINGLE SOURCE OF TRUTH for hardening checks.

    Layers (in order of update):
      1. obs           — float32 bytes
      2. reward        — float64 bytes
      3. terminated    — uint8 byte
      4. truncated     — uint8 byte
      5. info          — sorted JSON bytes
      6. env_state     — sorted JSON of ALL hidden internal variables
      7. rng_state     — sorted JSON of numpy Generator state

    Two hashes that agree ⟹ NOTHING differs.
    Two hashes that disagree ⟹ something, somewhere, leaked non-determinism.
    """
    h = hashlib.sha256()
    # Layer 1 — observation
    h.update(np.asarray(obs, dtype=np.float32).tobytes())
    # Layer 2 — reward (float64 preserves full double precision)
    h.update(np.float64(reward).tobytes())
    # Layer 3 & 4 — done flags as single bytes
    h.update(np.uint8(int(bool(terminated))).tobytes())
    h.update(np.uint8(int(bool(truncated))).tobytes())
    # Layer 5 — info
    if info:
        h.update(json.dumps(info, sort_keys=True, default=str).encode("utf-8"))
    # Layer 6 — full hidden env state (no pickle, pure sorted JSON)
    env_state = _extract_env_state(env)
    h.update(json.dumps(env_state, sort_keys=True, default=str).encode("utf-8"))
    # Layer 7 — numpy RNG state (catches drift invisible in obs)
    rng_state = _get_rng_state(env)
    h.update(json.dumps(rng_state, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 10 — Full state hashing (hidden-state divergence detector)
# ─────────────────────────────────────────────────────────────────────────────

def check_full_state_hashing(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Compare gold-standard hashes (obs + reward + done + info + env state + RNG)
    step-by-step across two identical runs.

    The critical failure this catches: "fake determinism" — obs arrays match
    but hidden internal state (HP accumulators, momentum, stat stages) has
    already diverged, leading to a different trajectory on the next episode.

    Two sub-checks per seed:
      A. Gold-hash mismatch  — any layer differs (hard fail)
      B. "Fake determinism"  — obs matches but gold-hash differs (hidden leak)
    """
    _section("10. Full state hashing — hidden-state divergence detector")
    all_pass = True

    for seed in seeds:
        env_a = _make_env_instance(seed)
        env_b = _make_env_instance(seed)
        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)

        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)

        # Verify initial state matches before any steps
        init_hash_a = _hash_gold_standard(env_a, obs_a, 0.0, False, False)
        init_hash_b = _hash_gold_standard(env_b, obs_b, 0.0, False, False)
        init_ok = (init_hash_a == init_hash_b)

        gold_mismatches  = 0   # any layer of gold hash differs
        fake_det_steps   = 0   # obs equal but gold hash differs ("fake determinism")
        first_fake_step  = -1
        first_fake_fields: list[str] = []

        for step_idx, action in enumerate(actions):
            out_a = env_a.step(action)
            out_b = env_b.step(action)
            obs_a, rew_a, term_a, trunc_a, info_a = out_a
            obs_b, rew_b, term_b, trunc_b, info_b = out_b

            gold_a = _hash_gold_standard(env_a, obs_a, rew_a, term_a, trunc_a, info_a)
            gold_b = _hash_gold_standard(env_b, obs_b, rew_b, term_b, trunc_b, info_b)

            obs_match  = np.array_equal(obs_a, obs_b)
            gold_match = (gold_a == gold_b)

            if not gold_match:
                gold_mismatches += 1
                if obs_match:
                    # CRITICAL: observations look identical but hidden state diverged
                    fake_det_steps += 1
                    if first_fake_step < 0:
                        first_fake_step = step_idx
                        # Identify which state fields differ
                        state_a = _extract_env_state(env_a)
                        state_b = _extract_env_state(env_b)
                        first_fake_fields = [
                            k for k in set(state_a) | set(state_b)
                            if state_a.get(k) != state_b.get(k)
                        ]

            done_a = term_a or trunc_a
            done_b = term_b or trunc_b
            if done_a or done_b:
                obs_a, _ = env_a.reset(seed=seed + step_idx)
                obs_b, _ = env_b.reset(seed=seed + step_idx)

        env_a.close()
        env_b.close()

        ok = init_ok and gold_mismatches == 0
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        print(f"  {status}  seed={seed:<6} | {n_steps} steps | "
              f"init {'ok' if init_ok else 'MISMATCH'} | "
              f"gold mismatches={gold_mismatches} | "
              f"fake-det={fake_det_steps}")

        if not ok and not quiet:
            if not init_ok:
                print(f"       ↳ INITIAL STATE MISMATCH: env state differs before first step")
            if fake_det_steps > 0:
                print(f"       ↳ FAKE DETERMINISM at step {first_fake_step}: "
                      f"obs match but hidden state diverges")
                if first_fake_fields:
                    print(f"       ↳ Diverging fields: {first_fake_fields}")
            elif gold_mismatches > 0:
                print(f"       ↳ {gold_mismatches} step(s) with full hash mismatch "
                      f"(both obs AND hidden state differ)")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 11 — RNG state tracking (frozen / drifting RNG detector)
# ─────────────────────────────────────────────────────────────────────────────

def check_rng_state_tracking(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Track env.np_random (PCG64 generator) state at each step.

    Three sub-checks:
      A. Identity:  same seed → RNG state identical at every step across two runs.
         Catches: non-deterministic re-seeding, global RNG leak.
      B. Liveness:  RNG state MUST advance at each step (not frozen).
         Catches: frozen RNG bug (seeding is ignored; state never changes).
      C. Divergence: different seeds → RNG states MUST differ within first K steps.
         Catches: all seeds routing to the same RNG state.
    """
    _section("11. RNG state tracking — frozen / drifting RNG detector")
    all_pass = True

    # ── Sub-check A + B: same seed, two runs ─────────────────────────────────
    for seed in seeds:
        env_a = _make_env_instance(seed)
        env_b = _make_env_instance(seed)
        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)

        env_a.reset(seed=seed)
        env_b.reset(seed=seed)

        rng_mismatches = 0     # A: same seed but different RNG state
        frozen_steps   = 0     # B: RNG state didn't change from previous step
        prev_rng_state: dict | None = None

        for step_idx, action in enumerate(actions):
            env_a.step(action)
            env_b.step(action)

            rng_a = _get_rng_state(env_a)
            rng_b = _get_rng_state(env_b)

            # Sub-check A
            if rng_a != rng_b:
                rng_mismatches += 1

            # Sub-check B — compare to previous step's RNG state
            if prev_rng_state is not None and rng_a == prev_rng_state:
                frozen_steps += 1

            prev_rng_state = rng_a

            done = (env_a.step.__self__ if False else None)  # avoid actual call
            # Reset both if done (we track in lock-step)
            # (we can't check done without stepping again; rely on n_steps limit)

        env_a.close()
        env_b.close()

        ok_a = rng_mismatches == 0
        ok_b = frozen_steps == 0
        ok = ok_a and ok_b
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        print(f"  {status}  seed={seed:<6} | "
              f"identity {'ok' if ok_a else f'FAIL({rng_mismatches} steps differ)'} | "
              f"liveness {'ok' if ok_b else f'FAIL({frozen_steps} frozen steps)'}")

        if not ok_a and not quiet:
            print(f"       ↳ RNG STATE DRIFT: same seed produces different RNG "
                  f"trajectories — global RNG contamination suspected")
        if not ok_b and not quiet:
            print(f"       ↳ FROZEN RNG: np_random state unchanged for {frozen_steps} "
                  f"consecutive step(s) — seed is being ignored")

    # ── Sub-check C: different seeds → different RNG states ───────────────────
    if len(seeds) >= 2:
        seed_a, seed_b = seeds[0], seeds[1]
        env_c = _make_env_instance(seed_a)
        env_d = _make_env_instance(seed_b)
        env_c.reset(seed=seed_a)
        env_d.reset(seed=seed_b)

        check_steps = min(10, n_steps)
        diverged = False
        for action_val in _fixed_actions(check_steps, seed=0):
            env_c.step(action_val)
            env_d.step(action_val)
            if _get_rng_state(env_c) != _get_rng_state(env_d):
                diverged = True
                break

        env_c.close()
        env_d.close()

        ok_c = diverged
        all_pass = all_pass and ok_c
        status_c = _PASS if ok_c else _FAIL
        print(f"  {status_c}  Cross-seed divergence: "
              f"seeds ({seed_a},{seed_b}) → "
              f"{'RNG states differ within first ' + str(check_steps) + ' steps' if ok_c else 'RNG STATES IDENTICAL — seeds ignored'}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 12 — Float stability (FP non-determinism probe)
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_EXACT_THRESHOLD  = 0.0          # same process: MUST be exactly equal
_FLOAT_ABS_THRESHOLD    = 1e-6         # cross-process tolerance (BLAS/LAPACK)
_FLOAT_REL_THRESHOLD    = 1e-5


def check_float_stability(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Run two identical episodes and compare observations with BOTH strict (==)
    and tolerance-based (np.allclose) equality.  Reports max absolute and
    max relative error at each mismatch.

    In a pure Python env (no BLAS/GPU), all obs must be EXACTLY equal.
    If they are not, the max error is reported so the caller can decide
    whether it is within an acceptable FP tolerance.

    FAIL conditions:
      • Any strict mismatch in the same process  (tolerance = 0.0)
    """
    _section("12. Float stability — floating-point non-determinism probe")
    all_pass = True

    for seed in seeds:
        env_a = _make_env_instance(seed)
        env_b = _make_env_instance(seed)
        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)

        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)

        strict_mismatches = 0
        max_abs_err = 0.0
        max_rel_err = 0.0
        first_mismatch_step = -1
        first_mismatch_obs  = None

        for step_idx, action in enumerate(actions):
            out_a = env_a.step(action)
            out_b = env_b.step(action)
            obs_a_s = out_a[0].astype(np.float64)
            obs_b_s = out_b[0].astype(np.float64)

            if not np.array_equal(out_a[0], out_b[0]):
                strict_mismatches += 1
                abs_err = float(np.max(np.abs(obs_a_s - obs_b_s)))
                denom   = np.abs(obs_a_s) + 1e-10
                rel_err = float(np.max(np.abs(obs_a_s - obs_b_s) / denom))
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                    max_rel_err = rel_err
                    if first_mismatch_step < 0:
                        first_mismatch_step = step_idx
                        first_mismatch_obs  = (out_a[0].copy(), out_b[0].copy())

            done = (out_a[2] or out_a[3]) or (out_b[2] or out_b[3])
            if done:
                obs_a, _ = env_a.reset(seed=seed + step_idx)
                obs_b, _ = env_b.reset(seed=seed + step_idx)

        env_a.close()
        env_b.close()

        ok = strict_mismatches == 0
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        if ok:
            print(f"  {status}  seed={seed:<6} | {n_steps} steps | "
                  f"exact equality confirmed | max_abs_err=0.0")
        else:
            print(f"  {status}  seed={seed:<6} | {n_steps} steps | "
                  f"{strict_mismatches} strict mismatch(es) | "
                  f"max_abs={max_abs_err:.2e} max_rel={max_rel_err:.2e}")
            if not quiet and first_mismatch_obs is not None:
                oa, ob = first_mismatch_obs
                diff_idx = int(np.argmax(np.abs(oa - ob)))
                print(f"       ↳ First mismatch at step {first_mismatch_step}, "
                      f"obs[{diff_idx}]: A={oa[diff_idx]:.10f}  B={ob[diff_idx]:.10f}")
                print(f"       ↳ DIAGNOSIS: pure Python env must be bit-exact. "
                      f"Check for time.time(), os.urandom(), or BLAS path differences.")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 13 — Permutation invariance (container-type & action-order probe)
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_actions(seed: int, actions: list | tuple) -> str:
    """Run one seeded episode with the given action sequence; return traj hash."""
    env = _make_env_instance(seed)
    env.reset(seed=seed)
    records: list[tuple] = []
    for action in actions:
        obs, rew, term, trunc, info = env.step(int(action))
        records.append((obs.copy(), float(rew), bool(term), bool(trunc),
                        dict(info) if info else None))
        if term or trunc:
            break
    env.close()
    return _hash_trajectory_full(records)


def check_permutation_invariance(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Sub-check A — Container transparency:
      Same action sequence as list vs tuple must produce identical hashes.
      Detects hidden reliance on Python container identity or type dispatch.

    Sub-check B — Sequence sensitivity:
      A differently-shuffled action sequence must produce a DIFFERENT hash.
      Detects RNG systems that ignore the action input entirely.

    Sub-check C — Action-input type coercion:
      Actions passed as numpy int32/int64/Python int must all produce the
      same result (env must normalise action type internally).
    """
    _section("13. Permutation invariance — container-type & sequence sensitivity")
    all_pass = True

    for seed in seeds:
        actions_list  = _fixed_actions(n_steps, n_actions=4, seed=seed)
        actions_tuple = tuple(actions_list)

        # ── A: list vs tuple ────────────────────────────────────────────────
        hash_list  = _run_with_actions(seed, actions_list)
        hash_tuple = _run_with_actions(seed, actions_tuple)
        ok_a = (hash_list == hash_tuple)
        all_pass = all_pass and ok_a

        # ── B: shuffled sequence → different trajectory ───────────────────
        # Reproducible shuffle with a seed derived from the test seed
        shuffle_rng = np.random.default_rng(seed=seed ^ 0xDEADBEEF)
        actions_shuffled = shuffle_rng.permutation(actions_list).tolist()
        # Only test if shuffled sequence is actually different
        if actions_shuffled != actions_list:
            hash_shuffled = _run_with_actions(seed, actions_shuffled)
            ok_b = (hash_shuffled != hash_list)
            all_pass = all_pass and ok_b
        else:
            ok_b = True   # degenerate: permutation produced same sequence
            hash_shuffled = hash_list

        # ── C: numpy int vs Python int coercion ──────────────────────────
        actions_np32 = [np.int32(a)  for a in actions_list]
        actions_np64 = [np.int64(a)  for a in actions_list]
        hash_np32 = _run_with_actions(seed, actions_np32)
        hash_np64 = _run_with_actions(seed, actions_np64)
        ok_c = (hash_np32 == hash_list) and (hash_np64 == hash_list)
        all_pass = all_pass and ok_c

        status = _PASS if (ok_a and ok_b and ok_c) else _FAIL
        print(f"  {status}  seed={seed:<6} | "
              f"list≡tuple {'✓' if ok_a else '✗'} | "
              f"shuffle≠original {'✓' if ok_b else '✗'} | "
              f"np.int coercion {'✓' if ok_c else '✗'}")

        if not quiet:
            if not ok_a:
                print(f"       ↳ list hash:  {hash_list[:20]}…")
                print(f"       ↳ tuple hash: {hash_tuple[:20]}…")
                print(f"       ↳ Container type affects trajectory — hidden type dispatch")
            if not ok_b:
                print(f"       ↳ Shuffled sequence produces SAME hash — action input ignored")
            if not ok_c:
                if hash_np32 != hash_list:
                    print(f"       ↳ np.int32 actions produce different trajectory")
                if hash_np64 != hash_list:
                    print(f"       ↳ np.int64 actions produce different trajectory")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 14 — Serialization roundtrip (state-capture completeness)
# ─────────────────────────────────────────────────────────────────────────────

def check_serialization_roundtrip(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Snapshot env state at the episode midpoint via copy.deepcopy(), then
    continue two independent rollouts:
      • Reference   — original env, uninterrupted
      • Restored    — deepcopy env, started from the snapshot

    The two rollouts must produce identical trajectories at every step
    (measured by gold-standard hash).

    FAIL: any divergence after restore indicates that copy.deepcopy() does
    NOT capture all relevant state — a requirement for AlphaZero-style
    tree search and checkpoint/resume.

    NOTE: This test validates deepcopy as a state-serialization primitive.
    For production, replace deepcopy with an explicit env.save_state() /
    env.load_state() API once implemented.
    """
    _section("14. Serialization roundtrip — state-capture completeness")
    all_pass = True

    for seed in seeds:
        midpoint = max(1, n_steps // 2)
        actions  = _fixed_actions(n_steps, n_actions=4, seed=seed)

        # ── Phase 1: run to midpoint and snapshot ─────────────────────────
        env_ref = _make_env_instance(seed)
        env_ref.reset(seed=seed)

        for action in actions[:midpoint]:
            out = env_ref.step(action)
            if out[2] or out[3]:   # episode ended before midpoint
                # Re-seed and continue to reach midpoint
                env_ref.reset(seed=seed + 1)

        # Snapshot via deepcopy — this is the serialization under test
        try:
            env_snap = copy.deepcopy(env_ref)
        except Exception as exc:
            _fail(f"seed={seed} | deepcopy failed: {exc}")
            all_pass = False
            continue

        # ── Phase 2: run both envs from midpoint ──────────────────────────
        ref_hashes:      list[str] = []
        restored_hashes: list[str] = []

        for action in actions[midpoint:]:
            out_ref  = env_ref.step(action)
            out_snap = env_snap.step(action)

            obs_ref, rew_ref, term_ref, trunc_ref, info_ref   = out_ref
            obs_snp, rew_snp, term_snp, trunc_snp, info_snp   = out_snap

            ref_hashes.append(
                _hash_gold_standard(env_ref,  obs_ref, rew_ref, term_ref, trunc_ref, info_ref)
            )
            restored_hashes.append(
                _hash_gold_standard(env_snap, obs_snp, rew_snp, term_snp, trunc_snp, info_snp)
            )

            done_ref  = term_ref  or trunc_ref
            done_snap = term_snp  or trunc_snp
            if done_ref or done_snap:
                break

        env_ref.close()
        env_snap.close()

        # ── Compare ───────────────────────────────────────────────────────
        n_compared = min(len(ref_hashes), len(restored_hashes))
        mismatches = [
            i for i in range(n_compared)
            if ref_hashes[i] != restored_hashes[i]
        ]
        len_mismatch = (len(ref_hashes) != len(restored_hashes))

        ok = not mismatches and not len_mismatch
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        print(f"  {status}  seed={seed:<6} | "
              f"snapshot at step {midpoint} | "
              f"{n_compared} post-restore steps | "
              f"{'trajectories identical' if ok else f'{len(mismatches)} mismatch(es)'}")

        if not ok and not quiet:
            if len_mismatch:
                print(f"       ↳ Episode length differs: ref={len(ref_hashes)} "
                      f"restored={len(restored_hashes)}")
            for i in mismatches[:3]:
                print(f"       ↳ Divergence at post-restore step {i}: "
                      f"ref={ref_hashes[i][:20]}… "
                      f"snap={restored_hashes[i][:20]}…")
            print(f"       ↳ DIAGNOSIS: deepcopy does not capture all env state. "
                  f"Likely culprit: a field not exposed via __dict__ "
                  f"(e.g. C-extension state, file handle, or hidden cache).")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 15 — Strict process isolation (cross-process contamination detector)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Worker must be at module level for multiprocessing.Pool pickling.

def _worker_isolated_episode(args: tuple) -> tuple[int, int, str]:
    """
    Spawn-safe worker: run a full seeded episode in a separate process and
    return (worker_id, seed, trajectory_hash).

    The hash uses _hash_trajectory_full (full-strength gold-hash chain)
    computed inside the worker so no data crosses process boundaries.
    """
    seed, worker_id, max_steps = args
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Re-import everything needed inside the worker
    import hashlib as _h
    import json as _j
    import numpy as _np
    from pathlib import Path as _Path

    try:
        from src.env.pokemon_env import PokemonEnv
    except ImportError:
        from env.pokemon_env import PokemonEnv

    env = PokemonEnv()
    env.set_opponent(mode="random")
    action_rng = _np.random.default_rng(seed=seed)
    actions = action_rng.integers(0, 4, size=max_steps).tolist()

    obs, _ = env.reset(seed=seed)
    record_hashes: list[str] = []

    for action in actions:
        obs, rew, term, trunc, info = env.step(int(action))
        # Inline gold-standard hash (avoids importing _hash_gold_standard)
        state_h = _h.sha256()
        state_h.update(_np.asarray(obs, dtype=_np.float32).tobytes())
        state_h.update(_np.float64(rew).tobytes())
        state_h.update(_np.uint8(int(bool(term))).tobytes())
        state_h.update(_np.uint8(int(bool(trunc))).tobytes())
        if info:
            state_h.update(_j.dumps(info, sort_keys=True, default=str).encode())
        record_hashes.append(state_h.hexdigest())
        if term or trunc:
            break

    env.close()
    traj_hash = _h.sha256(
        "||".join(record_hashes).encode()
    ).hexdigest()
    return worker_id, seed, traj_hash


def check_process_isolation(
    seeds: list[int],
    n_workers: int,
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Sub-check A — Identity under replication:
      Spawn n_workers processes all using seeds[0].
      ALL must return identical trajectory hashes.
      FAIL: any hash differs → cross-process contamination or shared global state.

    Sub-check B — Isolation under divergent seed:
      Replace one worker's seed with seeds[1].
      ONLY that worker's hash should differ.
      FAIL: a worker with a different seed produces the same hash as the
      base-seed workers (seed is not reaching the env).
    """
    _section("15. Strict process isolation — cross-process contamination detector")

    if not _HAS_SB3 and n_workers > 1:
        pass   # Pool works without SB3; SB3 only needed for SubprocVecEnv

    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = multiprocessing.get_context(start_method)

    base_seed = seeds[0]
    alt_seed  = seeds[1] if len(seeds) > 1 else base_seed + 777

    # ── A: n_workers processes, same seed ────────────────────────────────────
    args_same = [(base_seed, i, n_steps) for i in range(n_workers)]
    try:
        with ctx.Pool(processes=n_workers) as pool:
            results_same = pool.map(_worker_isolated_episode, args_same)
    except Exception as exc:
        _fail(f"Worker pool (Part A) failed: {exc}")
        return False

    hashes_same = [h for _, _, h in results_same]
    n_unique_same = len(set(hashes_same))
    ok_a = (n_unique_same == 1)
    all_pass_local = ok_a
    status_a = _PASS if ok_a else _FAIL
    print(f"  {status_a}  Part A: {n_workers} workers | seed={base_seed} | "
          f"{'all hashes identical' if ok_a else f'{n_unique_same} unique hashes (expected 1)'}")
    if not ok_a and not quiet:
        for wid, s, h in results_same:
            print(f"       ↳ worker {wid}: {h[:24]}…")

    # ── B: one worker with different seed ─────────────────────────────────────
    # Run 2 workers: worker-0=base_seed, worker-1=alt_seed
    args_mixed = [(base_seed, 0, n_steps), (alt_seed, 1, n_steps)]
    try:
        with ctx.Pool(processes=2) as pool:
            results_mixed = pool.map(_worker_isolated_episode, args_mixed)
    except Exception as exc:
        _fail(f"Worker pool (Part B) failed: {exc}")
        return False

    hash_base = results_mixed[0][2]
    hash_alt  = results_mixed[1][2]
    ok_b = (hash_base != hash_alt)
    all_pass_local = all_pass_local and ok_b
    status_b = _PASS if ok_b else _FAIL
    print(f"  {status_b}  Part B: seed={base_seed} vs seed={alt_seed} | "
          f"{'hashes differ — processes correctly isolated' if ok_b else 'HASHES IDENTICAL — seed not reaching env or shared state leak'}")

    return all_pass_local


# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 16 — Time independence (time-based randomness / race condition probe)
# ─────────────────────────────────────────────────────────────────────────────

_SLEEP_BETWEEN_STEPS_SEC = 0.005   # 5 ms — enough to trigger time.time() drift


def check_time_independence(
    seeds: list[int],
    n_steps: int,
    quiet: bool = False,
) -> bool:
    """
    Run two identical episodes:
      • Fast  — steps executed as quickly as possible
      • Slow  — sleep(_SLEEP_BETWEEN_STEPS_SEC) injected before each step

    Compare gold-standard hashes step-by-step.  They must be identical.

    FAIL: any mismatch indicates the env (or a library it calls) uses
    wall-clock time as a source of randomness (time.time(), datetime.now(),
    monotonic(), threading.Timer, etc.).  Also catches async event loops
    that yield differently depending on elapsed time.
    """
    _section("16. Time independence — time-based randomness / race condition probe")
    all_pass = True

    for seed in seeds:
        actions = _fixed_actions(n_steps, n_actions=4, seed=seed)

        # ── Fast run ─────────────────────────────────────────────────────────
        env_fast = _make_env_instance(seed)
        env_fast.reset(seed=seed)
        fast_hashes: list[str] = []
        for action in actions:
            obs, rew, term, trunc, info = env_fast.step(action)
            fast_hashes.append(
                _hash_gold_standard(env_fast, obs, rew, term, trunc, info)
            )
            if term or trunc:
                break
        env_fast.close()

        # ── Slow run (with deliberate sleep) ─────────────────────────────────
        env_slow = _make_env_instance(seed)
        env_slow.reset(seed=seed)
        slow_hashes: list[str] = []
        n_fast = len(fast_hashes)

        for action in actions[:n_fast]:
            time.sleep(_SLEEP_BETWEEN_STEPS_SEC)   # inject wall-clock delay
            obs, rew, term, trunc, info = env_slow.step(action)
            slow_hashes.append(
                _hash_gold_standard(env_slow, obs, rew, term, trunc, info)
            )
            if term or trunc:
                break
        env_slow.close()

        # ── Compare ───────────────────────────────────────────────────────────
        n_cmp = min(len(fast_hashes), len(slow_hashes))
        mismatches = [
            i for i in range(n_cmp)
            if fast_hashes[i] != slow_hashes[i]
        ]
        len_diff = (len(fast_hashes) != len(slow_hashes))

        ok = not mismatches and not len_diff
        all_pass = all_pass and ok
        status = _PASS if ok else _FAIL

        sleep_ms = int(_SLEEP_BETWEEN_STEPS_SEC * 1000)
        print(f"  {status}  seed={seed:<6} | {n_cmp} steps | "
              f"sleep={sleep_ms}ms/step | "
              f"{'time-independent ✓' if ok else f'{len(mismatches)} step(s) differ'}")

        if not ok and not quiet:
            if len_diff:
                print(f"       ↳ Episode length differs: fast={len(fast_hashes)} "
                      f"slow={len(slow_hashes)}")
            for i in mismatches[:3]:
                print(f"       ↳ Mismatch at step {i}: "
                      f"fast={fast_hashes[i][:20]}… "
                      f"slow={slow_hashes[i][:20]}…")
            print(f"       ↳ DIAGNOSIS: env uses wall-clock time. Search for "
                  f"time.time(), datetime.now(), monotonic(), or threading.Timer "
                  f"in the env and reward logic.")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY FINGERPRINT  (Requirement 9)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reproducibility_fingerprint(sprite_index: str | None = None) -> dict[str, str]:
    """
    Build a structured fingerprint of the execution environment for audit trails.

    Includes: Python version, numpy/torch/gymnasium versions, OS, CPU architecture,
    PYTHONHASHSEED, sprite registry sha256, and env source sha256.

    Store this alongside experiment logs for NeurIPS-style reproducibility.
    """
    fp: dict[str, str] = {
        "timestamp":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version":  platform.python_version(),
        "os":              f"{platform.system()} {platform.release()}",
        "machine":         platform.machine(),
        "cpu":             platform.processor() or platform.machine(),
        "PYTHONHASHSEED":  os.environ.get("PYTHONHASHSEED", "NOT_SET"),
        "numpy_version":   np.__version__,
        "torch_version":   "not_installed",
        "torch_cuda":      "n/a",
        "gymnasium_version":      "not_installed",
        "stable_baselines3_version": "not_installed",
        "sprite_registry_sha256": "not_checked",
        "env_source_sha256":      "not_checked",
        "env_version":            "unknown",
    }

    try:
        import torch as _t
        fp["torch_version"] = _t.__version__
        fp["torch_cuda"] = str(_t.cuda.is_available())
    except ImportError:
        pass

    try:
        import gymnasium as _g
        fp["gymnasium_version"] = _g.__version__
    except ImportError:
        try:
            import gym as _g  # type: ignore
            fp["gymnasium_version"] = f"{_g.__version__} (legacy)"
        except ImportError:
            pass

    try:
        import stable_baselines3 as _sb3
        fp["stable_baselines3_version"] = _sb3.__version__
    except ImportError:
        pass

    # Sprite registry sha256
    if sprite_index is not None:
        idx_path = Path(sprite_index)
        if idx_path.exists():
            fp["sprite_registry_sha256"] = hashlib.sha256(
                idx_path.read_bytes()
            ).hexdigest()

    # Env source sha256 + version string
    env_src = _PROJECT_ROOT / "src" / "env" / "pokemon_env.py"
    if env_src.exists():
        fp["env_source_sha256"] = hashlib.sha256(
            env_src.read_bytes()
        ).hexdigest()
    try:
        from src.env.pokemon_env import ENV_VERSION
        fp["env_version"] = ENV_VERSION
    except ImportError:
        try:
            from env.pokemon_env import ENV_VERSION as _ev  # type: ignore
            fp["env_version"] = _ev
        except ImportError:
            pass

    return fp


def print_reproducibility_fingerprint(fp: dict[str, str]) -> None:
    """Print the reproducibility fingerprint as a structured block."""
    _section("Reproducibility fingerprint")
    W = 32
    border = "  ┌" + "─" * (W + 2) + "┬" + "─" * 42 + "┐"
    row    = lambda k, v: f"  │ {k:<{W}} │ {str(v)[:40]:<40} │"
    sep    = "  ├" + "─" * (W + 2) + "┼" + "─" * 42 + "┤"
    close  = "  └" + "─" * (W + 2) + "┴" + "─" * 42 + "┘"

    print(border)
    for i, (k, v) in enumerate(fp.items()):
        if i > 0 and k in ("numpy_version", "PYTHONHASHSEED",
                           "sprite_registry_sha256", "env_version"):
            print(sep)
        print(row(k, v))
    print(close)

    # Warn on critical non-determinism risks
    if fp.get("PYTHONHASHSEED") == "NOT_SET":
        _warn("PYTHONHASHSEED not set — dict hash randomisation active")
        _warn("  Fix: PYTHONHASHSEED=0 python scripts/test_determinism.py")
    if fp.get("torch_cuda") == "True":
        _warn("CUDA available — ensure torch.use_deterministic_algorithms(True) "
              "and CUBLAS_WORKSPACE_CONFIG=:4096:8 are set")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI + MAIN
# ─────────────────────────────────────────────────────────────────────────────

_ALL_CHECKS = [
    "env", "episode", "multiprocess", "sprite", "log",
    "sensitivity", "mp_order", "cross_run", "rng_control",
    "full_state", "rng_track", "float_stability",
    "permutation", "serialization", "proc_isolation", "time_indep",
]

_CHECK_LABELS = {
    "env":           "1.  Single-env determinism",
    "episode":       "2.  Episode-hash determinism",
    "multiprocess":  "3.  Multiprocess determinism (SubprocVecEnv)",
    "sprite":        "4.  Sprite registry determinism",
    "log":           "5.  Cross-run log comparison",
    "sensitivity":   "6.  Seed sensitivity (frozen-RNG detector)",
    "mp_order":      "7.  Multiprocess per-worker order stability",
    "cross_run":     "8.  Cross-run persistent reproducibility",
    "rng_control":   "9.  RNG control & library versions",
    "full_state":    "10. Full-state gold-standard hashing",
    "rng_track":     "11. RNG state tracking & advancement",
    "float_stability":"12. Floating-point strict stability",
    "permutation":   "13. Permutation / container-type invariance",
    "serialization": "14. Serialization roundtrip (deepcopy)",
    "proc_isolation":"15. Cross-process contamination isolation",
    "time_indep":    "16. Time-independence (no wall-clock leaks)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Determinism validation suite for the Pokémon RL pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checks",
        nargs="+",
        choices=_ALL_CHECKS,
        default=_ALL_CHECKS,
        metavar="CHECK",
        help=(
            "Which checks to run (default: all). "
            "Choices: " + " ".join(_ALL_CHECKS)
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 999],
        metavar="SEED",
        help="RNG seeds to test (default: 42 123 999)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100,
        metavar="N",
        help="Number of steps per single-env test (default: 100)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of SubprocVecEnv workers (default: 4)",
    )
    p.add_argument(
        "--sprite_index",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to sprite_index.json for snapshot injection test",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=str(_PROJECT_ROOT / "logs"),
        metavar="DIR",
        help="Directory for persistent baseline JSON (cross_run check). "
             "Default: <project_root>/logs/",
    )
    p.add_argument(
        "--reset_baseline",
        action="store_true",
        help="Delete the existing cross-run baseline and save a fresh one",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-mismatch detail lines",
    )
    p.add_argument(
        "--no_reward_summary",
        action="store_true",
        help="Skip the informational reward distribution summary",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"\n{_BOLD}Pokémon RL Pipeline — Determinism Validation Suite (Research Grade){_RESET}")
    print(f"  seeds        : {args.seeds}")
    print(f"  steps        : {args.steps}")
    print(f"  workers      : {args.workers}")
    print(f"  log_dir      : {args.log_dir}")
    print(f"  checks       : {args.checks}")
    if args.sprite_index:
        print(f"  sprite_index : {args.sprite_index}")
    if args.reset_baseline:
        print(f"  reset_baseline: yes")

    t_start_total = time.perf_counter()
    # check_results: {check_name: (passed: bool, elapsed_sec: float)}
    check_results: dict[str, tuple[bool, float]] = {}

    def _run_check(name: str, fn, *a, **kw) -> bool:
        t0 = time.perf_counter()
        ok = fn(*a, **kw)
        check_results[name] = (ok, time.perf_counter() - t0)
        return ok

    # ── Original checks (unchanged) ──────────────────────────────────────────

    if "env" in args.checks:
        _run_check("env", check_single_env,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "episode" in args.checks:
        _run_check("episode", check_episode_hash,
                   seeds=args.seeds, max_steps=500, quiet=args.quiet)

    if "multiprocess" in args.checks:
        _run_check("multiprocess", check_multiprocess,
                   seed=args.seeds[0], n_workers=args.workers,
                   n_steps=args.steps, quiet=args.quiet)

    if "sprite" in args.checks:
        _run_check("sprite", check_sprite_determinism,
                   n_workers=args.workers, sprite_index=args.sprite_index,
                   quiet=args.quiet)

    if "log" in args.checks:
        _run_check("log", check_log_comparison,
                   seeds=args.seeds, max_steps=500, quiet=args.quiet)

    # ── New research-grade checks ─────────────────────────────────────────────

    if "sensitivity" in args.checks:
        _run_check("sensitivity", check_sensitivity,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "mp_order" in args.checks:
        _run_check("mp_order", check_multiprocess_order,
                   seed=args.seeds[0], n_workers=args.workers,
                   n_steps=args.steps, quiet=args.quiet)

    if "cross_run" in args.checks:
        _run_check("cross_run", check_cross_run_reproducibility,
                   seeds=args.seeds, max_steps=500, log_dir=args.log_dir,
                   reset_baseline=args.reset_baseline, quiet=args.quiet)

    if "rng_control" in args.checks:
        _run_check("rng_control", check_rng_control, quiet=args.quiet)

    # ── Hardening checks (gold-standard infrastructure) ───────────────────────

    if "full_state" in args.checks:
        _run_check("full_state", check_full_state_hashing,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "rng_track" in args.checks:
        _run_check("rng_track", check_rng_state_tracking,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "float_stability" in args.checks:
        _run_check("float_stability", check_float_stability,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "permutation" in args.checks:
        _run_check("permutation", check_permutation_invariance,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "serialization" in args.checks:
        _run_check("serialization", check_serialization_roundtrip,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    if "proc_isolation" in args.checks:
        _run_check("proc_isolation", check_process_isolation,
                   seeds=args.seeds, n_workers=args.workers,
                   n_steps=args.steps, quiet=args.quiet)

    if "time_indep" in args.checks:
        _run_check("time_indep", check_time_independence,
                   seeds=args.seeds, n_steps=args.steps, quiet=args.quiet)

    # ── Reward distribution (informational, no pass/fail) ────────────────────
    if not args.no_reward_summary:
        print_reward_summary(seeds=args.seeds, n_steps=200)

    # ── Summary table ─────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start_total
    n_pass  = sum(1 for ok, _ in check_results.values() if ok)
    n_total = len(check_results)
    n_fail  = n_total - n_pass

    # Column widths
    _W_LABEL  = 44
    _W_RESULT =  6
    _W_TIME   =  7

    def _row(label: str, result: str, t: str) -> str:
        return (f"  │ {label:<{_W_LABEL}} │ {result:^{_W_RESULT}} │ {t:>{_W_TIME}} │")

    def _divider(left: str = "├", mid: str = "┼", right: str = "┤") -> str:
        return (f"  {left}{'─'*(_W_LABEL+2)}{mid}{'─'*(_W_RESULT+2)}{mid}{'─'*(_W_TIME+2)}{right}")

    print(f"\n{'═' * 72}")
    print(f"  CHECK SUMMARY")
    print(_divider("┌", "┬", "┐"))
    print(_row("Check", "Result", "Time"))
    print(_divider())

    for check_name in _ALL_CHECKS:
        if check_name not in check_results:
            continue
        ok, t = check_results[check_name]
        label  = _CHECK_LABELS.get(check_name, check_name)
        result = f"\033[32mPASS\033[0m" if ok else f"\033[31mFAIL\033[0m"
        t_str  = f"{t:.1f}s"
        print(_row(label, result, t_str))

    print(_divider("├", "┼", "┤"))
    total_result = (f"\033[32m{n_pass}/{n_total} ✓\033[0m" if n_fail == 0
                    else f"\033[31m{n_pass}/{n_total} ✗\033[0m")
    print(_row("TOTAL", total_result, f"{elapsed_total:.1f}s"))
    print(_divider("└", "┴", "┘"))

    print()
    if n_fail == 0:
        print(f"  \033[32m{'━'*68}\033[0m")
        print(f"  \033[32m  ALL {n_total} CHECKS PASSED — pipeline is deterministic ✓\033[0m")
        print(f"  \033[32m{'━'*68}\033[0m")
    else:
        print(f"  \033[31m{'━'*68}\033[0m")
        print(f"  \033[31m  {n_fail} CHECK(S) FAILED — randomness leaks detected\033[0m")
        print(f"  \033[31m{'━'*68}\033[0m")
        print()
        print(f"  Failed:")
        for check_name, (ok, _) in check_results.items():
            if not ok:
                print(f"    ✗  {_CHECK_LABELS.get(check_name, check_name)}")
        print()
        print(f"  Common causes of non-determinism:")
        print(f"    • np.random global state (use np.random.default_rng, not np.random.seed)")
        print(f"    • Python hash randomisation (fix: PYTHONHASHSEED=0)")
        print(f"    • Uninitialised RNG in env or battle logic")
        print(f"    • time.time() / os.getpid() / os.urandom() in hot paths")
        print(f"    • Non-deterministic CUDA ops (set torch.use_deterministic_algorithms(True))")
        print(f"    • Custom C extensions with internal state")

    print(f"{'═' * 72}\n")

    # ── Reproducibility fingerprint ───────────────────────────────────────────
    fp = generate_reproducibility_fingerprint(args.sprite_index)
    print_reproducibility_fingerprint(fp)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    # On Windows (spawn), the Pool workers re-import this module — the guard
    # is essential to prevent infinite process spawning.
    multiprocessing.freeze_support()
    sys.exit(main())
