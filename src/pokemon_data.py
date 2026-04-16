"""
pokemon_data.py
~~~~~~~~~~~~~~~
Clean Pokémon data builders — one per layer.

build_pokemon_for_training(template) → dict
    ┌─────────────────────────────────────────────────────────┐
    │  TRAINING LAYER ONLY.  Returns a deep-copied, fully     │
    │  initialised Pokémon dict from TRAINING_ROSTER.         │
    │  Must NEVER be called from UI code.                     │
    │  Must NEVER change without bumping ENV_VERSION.         │
    └─────────────────────────────────────────────────────────┘

build_pokemon_for_ui(api_data) → dict
    ┌─────────────────────────────────────────────────────────┐
    │  UI LAYER ONLY.  Normalises a PokeAPI response dict for │
    │  use in the Streamlit dashboard and BattleEngine.       │
    │  Must NEVER be called from training code.               │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy

# ──────────────────────────────────────────────────────────────────────────
# TRAINING LAYER
# ──────────────────────────────────────────────────────────────────────────


def build_pokemon_for_training(template: dict) -> dict:
    """
    Build a deterministic, initialised Pokémon dict for PPO training.

    Parameters
    ----------
    template : dict
        One entry from TRAINING_ROSTER in pokemon_env.py.

    Returns
    -------
    dict — a deep copy with all runtime fields initialised to
           training-start defaults.

    This function is the SINGLE authoritative way to create Pokémon for
    the training environment.  DO NOT add fields here without also
    checking whether they affect OBSERVATION_SHAPE or ACTION_SIZE.
    """
    pokemon = copy.deepcopy(template)
    pokemon["stats"]       = dict(pokemon["base_stats"])
    pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    pokemon["current_hp"]  = 1.0
    pokemon["status"]      = None
    pokemon["item"]        = None
    pokemon["debilitado"]  = False
    return pokemon


def get_training_roster_pokemon(name: str) -> dict | None:
    """
    Return a fresh training dict for a Pokémon from the TRAINING_ROSTER.

    Returns None if the name is not found.
    """
    try:
        from src.env.pokemon_env import TRAINING_ROSTER
    except ImportError:
        from env.pokemon_env import TRAINING_ROSTER

    for template in TRAINING_ROSTER:
        if template["name"].lower() == name.lower():
            return build_pokemon_for_training(template)
    return None


# ──────────────────────────────────────────────────────────────────────────
# UI / GAME ENGINE LAYER
# ──────────────────────────────────────────────────────────────────────────


def build_pokemon_for_ui(api_data: dict) -> dict:
    """
    Prepare a PokeAPI-sourced Pokémon dict for the dashboard and
    BattleEngine.  Ensures all required runtime fields are present.

    This function MUST NOT be called from any training script.

    The moveset is expected to have been applied BEFORE calling this
    function (e.g. via competitive_movesets.build_moveset).  This
    function only guarantees structural completeness.
    """
    try:
        from src.battle_utils import apply_stat_stages
    except ImportError:
        from battle_utils import apply_stat_stages

    pokemon = dict(api_data)   # shallow copy — moves/pool are already set

    # Ensure all required runtime fields exist
    if "stat_stages" not in pokemon:
        pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    if "stats" not in pokemon or not pokemon["stats"]:
        pokemon["stats"] = dict(pokemon.get("base_stats", {}))
    if "current_hp" not in pokemon:
        pokemon["current_hp"] = 1.0
    if "status" not in pokemon:
        pokemon["status"] = None
    if "debilitado" not in pokemon:
        pokemon["debilitado"] = False

    # Recompute stats from base_stats + stages to guarantee consistency
    pokemon["stats"] = apply_stat_stages(pokemon["base_stats"], pokemon["stat_stages"])
    return pokemon


def reset_pokemon_battle_state(pokemon: dict) -> None:
    """
    Reset battle-runtime fields in-place so a Pokémon is ready for a
    fresh battle.  Used by the dashboard's 'INICIAR COMBATE' handler.
    """
    try:
        from src.battle_utils import apply_stat_stages
    except ImportError:
        from battle_utils import apply_stat_stages

    pokemon["current_hp"]  = 1.0
    pokemon["status"]      = None
    pokemon["debilitado"]  = False
    pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    pokemon["stats"]       = apply_stat_stages(pokemon["base_stats"], pokemon["stat_stages"])
