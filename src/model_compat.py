import json
import os
from dataclasses import dataclass

from stable_baselines3 import PPO

try:
    from src.env.pokemon_env import ACTION_SIZE, ENV_VERSION, OBSERVATION_SHAPE
except ImportError:
    from env.pokemon_env import ACTION_SIZE, ENV_VERSION, OBSERVATION_SHAPE


@dataclass
class ModelCompatibility:
    is_valid: bool
    reason: str
    obs_shape: tuple | None = None
    action_n: int | None = None
    env_version: str | None = None


def _meta_path(model_base_path):
    return f"{model_base_path}.meta.json"


def save_model_metadata(model_base_path):
    meta = {
        "env_version": ENV_VERSION,
        "observation_shape": list(OBSERVATION_SHAPE),
        "action_n": ACTION_SIZE,
    }
    with open(_meta_path(model_base_path), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def check_model_compatibility(model_base_path):
    try:
        model = PPO.load(model_base_path)
    except Exception as exc:
        return ModelCompatibility(False, f"load_failed: {exc}")

    obs_shape = getattr(model.observation_space, "shape", None)
    action_n = getattr(model.action_space, "n", None)
    if tuple(obs_shape or ()) != tuple(OBSERVATION_SHAPE):
        return ModelCompatibility(
            False,
            f"legacy_incompatible_obs: expected {OBSERVATION_SHAPE}, found {obs_shape}",
            obs_shape=obs_shape,
            action_n=action_n,
        )
    if action_n != ACTION_SIZE:
        return ModelCompatibility(
            False,
            f"legacy_incompatible_action: expected Discrete({ACTION_SIZE}), found {action_n}",
            obs_shape=obs_shape,
            action_n=action_n,
        )

    meta_file = _meta_path(model_base_path)
    if not os.path.exists(meta_file):
        return ModelCompatibility(
            False,
            "legacy_incompatible_metadata: missing env metadata file",
            obs_shape=obs_shape,
            action_n=action_n,
        )

    try:
        with open(meta_file, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        return ModelCompatibility(
            False,
            f"legacy_incompatible_metadata: cannot read metadata ({exc})",
            obs_shape=obs_shape,
            action_n=action_n,
        )

    meta_env_version = meta.get("env_version")
    meta_obs_shape = tuple(meta.get("observation_shape", []))
    meta_action = meta.get("action_n")
    if meta_env_version != ENV_VERSION or meta_obs_shape != tuple(OBSERVATION_SHAPE) or meta_action != ACTION_SIZE:
        return ModelCompatibility(
            False,
            "legacy_incompatible_metadata: env contract mismatch",
            obs_shape=obs_shape,
            action_n=action_n,
            env_version=meta_env_version,
        )

    return ModelCompatibility(True, "compatible", obs_shape=obs_shape, action_n=action_n, env_version=meta_env_version)


def require_compatible_model(model_base_path):
    compat = check_model_compatibility(model_base_path)
    if not compat.is_valid:
        raise RuntimeError(f"Model '{model_base_path}' is LEGACY - INCOMPATIBLE ({compat.reason})")
    return PPO.load(model_base_path)
