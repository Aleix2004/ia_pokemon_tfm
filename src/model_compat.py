import json
import os
import warnings
from dataclasses import dataclass, field

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
    # Holds the already-loaded PPO model so require_compatible_model
    # can reuse it instead of calling PPO.load() a second time.
    model: object = field(default=None, repr=False, compare=False)


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

    # Metadata is advisory: auto-generate if missing instead of rejecting the model.
    # A model whose spaces already match the env contract IS compatible regardless of
    # whether save_model_metadata() was ever called after training.
    meta_file = _meta_path(model_base_path)
    if not os.path.exists(meta_file):
        try:
            save_model_metadata(model_base_path)
            warnings.warn(
                f"[model_compat] Auto-generated missing metadata for '{model_base_path}'. "
                "Run save_model_metadata() after training to avoid this.",
                UserWarning,
                stacklevel=2,
            )
        except OSError:
            # Read-only filesystem (e.g. mounted volume) — not a blocker.
            pass

    # Cross-check the metadata version; warn but do not reject if it drifts.
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("env_version") != ENV_VERSION:
                warnings.warn(
                    f"[model_compat] env_version mismatch in metadata for "
                    f"'{model_base_path}': file='{meta.get('env_version')}' "
                    f"current='{ENV_VERSION}'. Spaces match — model is still usable.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass  # Corrupt / unreadable metadata — spaces already validated above.

    return ModelCompatibility(
        True, "compatible",
        obs_shape=obs_shape,
        action_n=action_n,
        env_version=ENV_VERSION,
        model=model,  # return the already-loaded instance
    )


def require_compatible_model(model_base_path):
    """Load, validate, and return a PPO model ready for inference.

    Raises RuntimeError if the model cannot be loaded or its spaces do not
    match the current env contract (OBSERVATION_SHAPE, ACTION_SIZE).
    Reuses the model instance already loaded during compatibility checking —
    no second PPO.load() call is made.
    """
    compat = check_model_compatibility(model_base_path)
    if not compat.is_valid:
        raise RuntimeError(
            f"Model '{model_base_path}' is incompatible: {compat.reason}"
        )
    return compat.model  # already loaded inside check_model_compatibility
