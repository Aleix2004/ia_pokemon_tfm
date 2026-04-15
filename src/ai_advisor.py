"""
ai_advisor.py
~~~~~~~~~~~~~
Hybrid PPO + heuristic action-selection layer.

The PPO model provides the base action. This module inspects the chosen move
and overrides it when the model makes a clearly suboptimal choice:

  1. Move has 0× effectiveness (type immunity) → always override.
  2. PPO's move has ≤ 0× expected damage AND a super-effective option
     exists → override with the best SE move.
  3. PPO's move is a status/0-power move when the opponent is already
     carrying that status condition → prefer a damaging move instead.

Nothing here changes the PPO model, its weights, observation shape, or
action space.  It is purely a post-processing filter for inference.
"""

from __future__ import annotations

try:
    from src.battle_utils import get_type_multiplier
except ImportError:
    from battle_utils import get_type_multiplier


# Minimum ratio: if the best available move has ≥ this many times the
# expected damage of PPO's choice AND is super-effective, we override.
_SE_OVERRIDE_RATIO = 2.0


def _move_expected_damage(move: dict, defender_types: list[str]) -> float:
    """
    Simple expected-damage score: power × type_effectiveness.
    Status moves (power == 0) return 0.
    """
    power = move.get("power") or 0
    effectiveness = get_type_multiplier(move.get("type"), defender_types)
    return float(power) * float(effectiveness)


def _score_all_moves(moves: list[dict], defender_types: list[str]) -> list[tuple[float, float, int]]:
    """
    Return a list of (expected_damage, effectiveness, index) for every move,
    sorted descending by expected_damage.
    """
    scored = []
    for idx, move in enumerate(moves):
        eff = get_type_multiplier(move.get("type"), defender_types)
        dmg = (move.get("power") or 0) * eff
        scored.append((dmg, eff, idx))
    scored.sort(reverse=True)
    return scored


def get_hybrid_action(
    ppo_action: int,
    ia_pokemon: dict,
    rival_pokemon: dict,
) -> int:
    """
    Return the best action index after filtering the PPO suggestion.

    Parameters
    ----------
    ppo_action:    Raw integer action from model.predict().
    ia_pokemon:    Active IA Pokémon dict (must have 'moves' and 'types').
    rival_pokemon: Active rival Pokémon dict (must have 'types').

    Returns
    -------
    int — final action to execute (0–3).
    """
    moves = ia_pokemon.get("moves") or []
    if not moves:
        return 0

    n_moves = len(moves)
    ppo_idx = max(0, min(int(ppo_action), n_moves - 1))
    defender_types = rival_pokemon.get("types") or []

    ppo_move = moves[ppo_idx]
    ppo_eff = get_type_multiplier(ppo_move.get("type"), defender_types)
    ppo_dmg = _move_expected_damage(ppo_move, defender_types)

    scored = _score_all_moves(moves, defender_types)

    # ── Rule 1: PPO chose a 0× (immune) move → always override ─────────────
    if ppo_eff == 0.0:
        for dmg, eff, idx in scored:
            if eff > 0.0:
                return idx
        # Every move is immune (unlikely) — return PPO's choice anyway
        return ppo_idx

    # ── Rule 2: PPO chose a status/0-damage move when opponent already has
    #    that status, and a damaging option exists ────────────────────────────
    rival_status = rival_pokemon.get("status")
    if ppo_dmg == 0 and rival_status:
        # Check if the status move would try to inflict the same condition
        try:
            from src.battle_mechanics import MOVE_STATUS_TABLE
        except ImportError:
            from battle_mechanics import MOVE_STATUS_TABLE
        raw_name = (ppo_move.get("api_name") or ppo_move.get("name") or "").lower().replace(" ", "-")
        entry = MOVE_STATUS_TABLE.get(raw_name)
        if entry and entry[0] == rival_status:
            # The status move is useless — pick the best damaging move instead
            for dmg, eff, idx in scored:
                if dmg > 0:
                    return idx

    # ── Rule 3: A super-effective move is ≥ _SE_OVERRIDE_RATIO × better ────
    if scored:
        best_dmg, best_eff, best_idx = scored[0]
        if (
            best_eff >= 2.0
            and best_idx != ppo_idx
            and ppo_dmg > 0   # PPO's move does something — just not as good
            and best_dmg >= ppo_dmg * _SE_OVERRIDE_RATIO
        ):
            return best_idx

    return ppo_idx


def get_greedy_action(ia_pokemon: dict, rival_pokemon: dict) -> int:
    """
    Pure greedy fallback: pick the move with the highest expected damage.
    Used when no PPO model is loaded.
    """
    moves = ia_pokemon.get("moves") or []
    if not moves:
        return 0
    defender_types = rival_pokemon.get("types") or []
    scored = _score_all_moves(moves, defender_types)
    # Prefer damaging moves; if all are status, just return index 0
    for dmg, eff, idx in scored:
        if dmg > 0:
            return idx
    return 0
