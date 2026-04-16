"""
obs_builder.py
~~~~~~~~~~~~~~
THE OBSERVATION CONTRACT shared between the training layer and the
game engine layer.

Both PokemonEnv._get_obs() (training) and BattleEngine._get_obs()
(inference) MUST produce identical outputs for identical inputs.
Centralising the formula here enforces that contract.

OBSERVATION_SHAPE = (28,)  — DO NOT CHANGE without bumping ENV_VERSION.

Layout (index → meaning):
  0      : me  current_hp (0-1)
  1      : foe current_hp (0-1)
  2-3    : me  type pair  (normalized type indices)
  4-5    : foe type pair
  6      : me  best-move type-effectiveness vs foe (normalized: 0=immune,
               0.125=0.5×, 0.25=neutral, 0.5=2×SE, 1.0=4×SE)
               → "how well can I hit you right now?"
  7      : foe best-move type-effectiveness vs me (same scale)
               → "how well can you hit me right now?"
  8-12   : me  stat stages  atk/def/sp_atk/sp_def/spd  (normalized)
  13-17  : foe stat stages
  18-22  : me  base stats   atk/def/sp_atk/sp_def/spd  (normalized /255)
  23-27  : foe base stats

NOTE — slots 6 & 7 semantic change (previously binary status flags):
  The training environment (PokemonEnv) never applies status conditions,
  so the old flags were ALWAYS 0.  PPO models trained under the old
  contract have near-zero weights for those positions, so repurposing
  them carries minimal inference impact while providing a rich switching
  signal for future training runs.
"""

import numpy as np

try:
    from src.battle_utils import TYPE_ORDER, get_type_index, get_type_multiplier
except ImportError:
    from battle_utils import TYPE_ORDER, get_type_index, get_type_multiplier

# ── Pure helper functions ──────────────────────────────────────────────────


def _type_pair(pokemon: dict) -> tuple[float, float]:
    types = pokemon.get("types", []) or ["normal"]
    n = max(1, len(TYPE_ORDER) - 1)
    t1 = get_type_index(types[0]) / n
    t2 = get_type_index(types[1]) / n if len(types) > 1 else 0.0
    return float(t1), float(t2)


def _best_eff_norm(me: dict, foe: dict) -> float:
    """
    Best type effectiveness of me's damaging moves against foe, normalised to [0, 1].

    Scale:
        0.000 = immune    (0×)
        0.063 = quarter   (0.25×)
        0.125 = resisted  (0.5×)
        0.250 = neutral   (1×)
        0.500 = super-eff (2×)
        1.000 = double-SE (4×)

    Falls back to neutral (0.25) when no damaging moves are present so the PPO
    receives a meaningful gradient even on pure-status movesets.
    """
    max_eff = max(
        (
            get_type_multiplier(m.get("type", "normal"), foe.get("types", []))
            for m in me.get("moves", [])
            if (m.get("power") or 0) > 0
        ),
        default=1.0,  # neutral fallback
    )
    return float(np.clip(max_eff / 4.0, 0.0, 1.0))


def _stage_norm(pokemon: dict, stat: str) -> float:
    """Stat stage in [-6, +6] normalised to [0, 1]."""
    return (pokemon.get("stat_stages", {}).get(stat, 0) + 6) / 12.0


def _stat_norm(pokemon: dict, stat: str) -> float:
    """Base stat normalised by 255 (max possible base stat)."""
    return min(1.0, pokemon.get("stats", {}).get(stat, 0) / 255.0)


# ── Public contract function ───────────────────────────────────────────────


def build_obs_28(me: dict, foe: dict) -> np.ndarray:
    """
    Build the 28-dimensional PPO observation vector from two Pokémon dicts.

    This function is the SOLE authoritative source for the observation
    format.  It is called by:
        PokemonEnv._get_obs()      (training layer)
        BattleEngine._get_obs()    (game engine layer)

    Parameters
    ----------
    me  : dict — the active Pokémon whose perspective this obs represents
    foe : dict — the opposing Pokémon

    Returns
    -------
    np.ndarray of shape (28,) and dtype float32
    """
    me_t1,  me_t2  = _type_pair(me)
    foe_t1, foe_t2 = _type_pair(foe)

    return np.array(
        [
            float(me.get("current_hp",  1.0)),
            float(foe.get("current_hp", 1.0)),
            me_t1,  me_t2,
            foe_t1, foe_t2,
            _best_eff_norm(me,  foe),   # slot 6: how well can ME  hit FOE?
            _best_eff_norm(foe, me),    # slot 7: how well can FOE hit ME?
            _stage_norm(me,  "atk"),
            _stage_norm(me,  "def"),
            _stage_norm(me,  "sp_atk"),
            _stage_norm(me,  "sp_def"),
            _stage_norm(me,  "spd"),
            _stage_norm(foe, "atk"),
            _stage_norm(foe, "def"),
            _stage_norm(foe, "sp_atk"),
            _stage_norm(foe, "sp_def"),
            _stage_norm(foe, "spd"),
            _stat_norm(me,  "atk"),
            _stat_norm(me,  "def"),
            _stat_norm(me,  "sp_atk"),
            _stat_norm(me,  "sp_def"),
            _stat_norm(me,  "spd"),
            _stat_norm(foe, "atk"),
            _stat_norm(foe, "def"),
            _stat_norm(foe, "sp_atk"),
            _stat_norm(foe, "sp_def"),
            _stat_norm(foe, "spd"),
        ],
        dtype=np.float32,
    )
