"""
ai_advisor.py
~~~~~~~~~~~~~
Hybrid PPO + heuristic action-selection layer.

The PPO model provides the base action. This module inspects the chosen move
and overrides it when the model makes a clearly suboptimal choice:

  1. Move has 0× effectiveness (type immunity) → always override.
  2. PPO chose a status/redundant move when a damaging option is clearly better.
  3. A super-effective move is significantly better than PPO's choice → override.
  4. (NEW) get_ia_switch_decision() — evaluates whether the IA should voluntarily
     switch to a better-matchup Pokémon before attacking.

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
# Lowered from 2.0 → 1.5 so the AI more aggressively picks SE moves.
_SE_OVERRIDE_RATIO = 1.5

# Matchup score threshold below which the IA considers switching.
# -0.25 means the rival has a significant type advantage AND the IA has
# at most neutral coverage — realistic but not hair-trigger.
_SWITCH_BAD_MATCHUP_THRESHOLD = -0.25

# Minimum matchup improvement a switch target must provide over the
# current active Pokémon to justify switching.
_SWITCH_MIN_IMPROVEMENT = 0.20

# If the rival's HP is below this fraction, finish them off instead of switching.
_SWITCH_RIVAL_HP_KILL_THRESHOLD = 0.30


def _compute_matchup(me: dict, foe: dict) -> float:
    """
    Signed matchup score in [-1, +1].
    Positive = me has type advantage; negative = foe has type advantage.
    """
    me_best = max(
        (
            get_type_multiplier(m.get("type", "normal"), foe.get("types", []))
            for m in me.get("moves", [])
            if (m.get("power") or 0) > 0
        ),
        default=1.0,
    )
    foe_best = max(
        (
            get_type_multiplier(m.get("type", "normal"), me.get("types", []))
            for m in foe.get("moves", [])
            if (m.get("power") or 0) > 0
        ),
        default=1.0,
    )
    return float((me_best - foe_best) / 4.0)


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

    # ── Rule 2b: PPO chose a status/0-damage move but we have a SE move ─────
    # Even without an existing status condition, if there is a 2× or 4× SE
    # damaging move available, prefer it over spamming status moves.
    if ppo_dmg == 0 and scored:
        best_dmg, best_eff, best_idx = scored[0]
        if best_eff >= 2.0 and best_dmg > 0:
            return best_idx

    # ── Rule 3: A super-effective move is ≥ _SE_OVERRIDE_RATIO × better ────
    if scored:
        best_dmg, best_eff, best_idx = scored[0]
        if (
            best_eff >= 2.0
            and best_idx != ppo_idx
            and best_dmg >= max(1.0, ppo_dmg) * _SE_OVERRIDE_RATIO
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


def get_ia_switch_decision(engine) -> tuple[bool, int | None]:
    """
    Decide whether the IA should voluntarily switch to a better-matchup Pokémon.

    This function is called before each IA turn. It evaluates the current
    type matchup and team state, and recommends switching when:

      • Current matchup is significantly bad  (matchup_score < threshold)
      • Rival is not nearly KO'd              (rival hp ≥ kill threshold)
      • A clearly better switch target exists (matchup improvement ≥ min_improvement)
      • The target is alive and not debilitated

    Parameters
    ----------
    engine : BattleEngine instance with get_state(), ia_pokemon, rival_pokemon.

    Returns
    -------
    (True, team_index)  — IA should switch to team[team_index]
    (False, None)       — IA should attack as normal
    """
    ia_pokemon    = getattr(engine, "ia_pokemon",    None)
    rival_pokemon = getattr(engine, "rival_pokemon", None)

    if not ia_pokemon or not rival_pokemon:
        return False, None

    # Don't switch if rival is nearly dead — finish them off
    rival_hp = float(rival_pokemon.get("current_hp", 1.0))
    if rival_hp < _SWITCH_RIVAL_HP_KILL_THRESHOLD:
        return False, None

    # Compute current matchup
    current_matchup = _compute_matchup(ia_pokemon, rival_pokemon)

    # Only consider switching when the matchup is clearly bad
    if current_matchup >= _SWITCH_BAD_MATCHUP_THRESHOLD:
        return False, None

    # Get the full IA team to find a better switch target
    try:
        state = engine.get_state()
    except Exception:
        return False, None

    team_ia    = state.get("team_ia", [])
    active_ia  = state.get("active_ia", 0)

    if len(team_ia) <= 1:
        return False, None

    best_idx   = None
    best_score = current_matchup  # only switch if the target is strictly better

    for idx, pokemon in enumerate(team_ia):
        if idx == active_ia:
            continue
        # Skip fainted / debilitated Pokémon
        if pokemon.get("debilitado", False):
            continue
        if float(pokemon.get("current_hp", 0.0)) <= 0.0:
            continue

        candidate_matchup = _compute_matchup(pokemon, rival_pokemon)

        # Must be a significant improvement over the current situation
        if candidate_matchup > best_score + _SWITCH_MIN_IMPROVEMENT:
            best_score = candidate_matchup
            best_idx   = idx

    if best_idx is not None:
        return True, best_idx

    return False, None
