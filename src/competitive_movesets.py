"""
competitive_movesets.py
~~~~~~~~~~~~~~~~~~~~~~~
Generates competitive 4-move sets for any Pokémon, following the same
role-based logic used in Pokémon Showdown team-building.

Public API
----------
build_moveset(name, types, base_stats, candidates, mode) → list[dict]
    Select the best 4 moves from a list of candidate move dicts.

prefilter_move_names(move_names, pokemon_types, limit) → list[str]
    Fast name-only pre-filter used BEFORE fetching PokeAPI move details,
    to keep the number of API calls manageable.

get_role_info(base_stats) → dict
    Classify a Pokémon's competitive role and return display metadata.

get_filtered_move_pool(candidates, types, base_stats, limit) → list[dict]
    Return the top-scored moves for a custom editing dropdown.

Constraints respected
---------------------
* OBSERVATION_SHAPE = (28,) — unchanged
* ACTION_SPACE = 4           — unchanged
* PPO training pipeline      — unaffected (pure data-layer module)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Type-effectiveness lookup
# attacking_type → frozenset of defending types it hits super-effectively
# ──────────────────────────────────────────────────────────────────────────

SE_AGAINST: dict[str, frozenset[str]] = {
    "normal":   frozenset(),
    "fire":     frozenset({"grass", "ice", "bug", "steel"}),
    "water":    frozenset({"fire", "ground", "rock"}),
    "electric": frozenset({"water", "flying"}),
    "grass":    frozenset({"water", "ground", "rock"}),
    "ice":      frozenset({"grass", "ground", "flying", "dragon"}),
    "fighting": frozenset({"normal", "ice", "rock", "dark", "steel"}),
    "poison":   frozenset({"grass", "fairy"}),
    "ground":   frozenset({"fire", "electric", "poison", "rock", "steel"}),
    "flying":   frozenset({"grass", "fighting", "bug"}),
    "psychic":  frozenset({"fighting", "poison"}),
    "bug":      frozenset({"grass", "psychic", "dark"}),
    "rock":     frozenset({"fire", "ice", "flying", "bug"}),
    "ghost":    frozenset({"ghost", "psychic"}),
    "dragon":   frozenset({"dragon"}),
    "dark":     frozenset({"ghost", "psychic"}),
    "steel":    frozenset({"ice", "rock", "fairy"}),
    "fairy":    frozenset({"fighting", "dragon", "dark"}),
}

# ──────────────────────────────────────────────────────────────────────────
# Move classification sets (api_name format: lowercase-hyphenated)
# ──────────────────────────────────────────────────────────────────────────

# Broadly useful competitive moves — get a score bonus in any moveset
_HIGH_PRIORITY: frozenset[str] = frozenset({
    # Physical finishers
    "earthquake", "close-combat", "flare-blitz", "brave-bird", "extreme-speed",
    "aqua-jet", "bullet-punch", "mach-punch", "ice-punch", "thunder-punch",
    "fire-punch", "stone-edge", "rock-slide", "iron-head", "meteor-mash",
    "knock-off", "sucker-punch", "crunch", "play-rough", "superpower",
    "cross-chop", "high-jump-kick", "poison-jab", "iron-tail", "zen-headbutt",
    "waterfall", "liquidation", "razor-shell", "night-slash", "slash",
    "shadow-claw", "phantom-force", "shadow-sneak", "wood-hammer", "power-whip",
    "leaf-blade", "x-scissor", "aerial-ace", "acrobatics", "drill-peck",
    "gyro-ball", "heavy-slam", "body-slam", "facade", "return",
    # Special finishers
    "thunderbolt", "ice-beam", "flamethrower", "surf", "psychic",
    "shadow-ball", "energy-ball", "sludge-bomb", "sludge-wave", "focus-blast",
    "aura-sphere", "earth-power", "flash-cannon", "dark-pulse", "moonblast",
    "dazzling-gleam", "boomburst", "hyper-voice", "scald", "draco-meteor",
    "dragon-pulse", "hydro-pump", "fire-blast", "blizzard", "thunder",
    "psyshock", "psystrike", "giga-drain", "leaf-storm", "power-gem",
    "discharge", "volt-switch", "signal-beam", "mystical-fire", "lava-plume",
    "air-slash", "hurricane", "aeroblast", "spacial-rend", "roar-of-time",
    "origin-pulse", "precipice-blades", "oblivion-wing", "judgment",
    # Physical setup
    "swords-dance", "dragon-dance", "bulk-up", "coil", "hone-claws",
    # Special setup
    "nasty-plot", "calm-mind", "quiver-dance", "tail-glow", "shell-smash",
    "work-up", "agility", "autotomize",
    # Status / support
    "toxic", "will-o-wisp", "thunder-wave", "leech-seed",
    "stealth-rock", "spikes", "rapid-spin", "defog",
    "recover", "roost", "soft-boiled", "slack-off", "synthesis",
    "moonlight", "morning-sun", "shore-up",
    "protect", "substitute", "u-turn", "flip-turn",
})

# Moves that should NEVER be in a competitive set
_EXCLUDED: frozenset[str] = frozenset({
    "splash", "transform", "conversion", "conversion-2", "celebrate",
    "hold-hands", "lock-on", "mind-reader", "me-first", "sketch",
    "mimic", "copycat", "assist", "metronome", "mirror-move",
    "tail-whip", "growl", "string-shot", "leer", "sand-attack",
    "smokescreen", "flash", "kinesis", "teleport",
    "bind", "wrap", "clamp", "constrict", "bide",
    "swagger", "flatter", "attract",
    "minimize", "double-team",
    "heal-bell", "aromatherapy", "perish-song",
    "trick-room", "wonder-room", "magic-room",
    "safeguard", "lucky-chant", "mist",
    "role-play", "gastro-acid", "worry-seed",
    "rage", "struggle",
})

# Functional category sets (api_name format)
_SETUP: frozenset[str] = frozenset({
    "swords-dance", "dragon-dance", "bulk-up", "coil", "hone-claws",
    "nasty-plot", "calm-mind", "quiver-dance", "tail-glow", "shell-smash",
    "work-up", "agility", "autotomize", "curse", "iron-defense",
    "amnesia", "acid-armor", "barrier", "stockpile",
})
_RECOVERY: frozenset[str] = frozenset({
    "recover", "roost", "soft-boiled", "slack-off", "synthesis",
    "moonlight", "morning-sun", "shore-up", "jungle-healing",
    "rest", "milk-drink", "wish", "heal-order",
})
_STATUS: frozenset[str] = frozenset({
    "toxic", "will-o-wisp", "thunder-wave", "glare",
    "spore", "sleep-powder", "hypnosis", "dark-void", "lovely-kiss",
    "leech-seed",
})
_PIVOT: frozenset[str] = frozenset({
    "u-turn", "volt-switch", "flip-turn", "teleport", "baton-pass",
})

# Keyword fragments used by the NAME-ONLY prefilter
_TYPE_KW: dict[str, frozenset[str]] = {
    "fire":     frozenset({"fire", "flam", "flare", "ember", "heat", "blaze", "lava", "inferno", "scorch", "mystical"}),
    "water":    frozenset({"water", "surf", "hydro", "aqua", "dive", "rain", "cascade", "stream", "liquidat", "waterfall", "scald"}),
    "electric": frozenset({"thunder", "volt", "electric", "spark", "charge", "bolt", "discharge", "nuzzle", "zap"}),
    "grass":    frozenset({"grass", "leaf", "seed", "petal", "solar", "energy-ball", "vine", "giga", "razor-leaf", "spore", "wood", "power-whip", "leaf-storm"}),
    "ice":      frozenset({"ice", "blizzard", "freeze", "frost", "cold", "aurora", "hail", "powder-snow", "avalanche"}),
    "fighting": frozenset({"punch", "kick", "combat", "chop", "hammer", "smash", "fist", "thrust", "superpower", "aura-sphere", "high-jump"}),
    "poison":   frozenset({"poison", "toxic", "sludge", "acid", "smog", "venoshock", "cross-poison"}),
    "ground":   frozenset({"earthquake", "earth", "mud", "drill", "dig", "sand-tomb", "precipice"}),
    "flying":   frozenset({"fly", "wing", "aerial", "air", "gust", "peck", "brave-bird", "roost", "feather", "hurricane", "oblivion"}),
    "psychic":  frozenset({"psychic", "psych", "mental", "extrasensory", "zen", "future", "psyshock", "psystrike", "spacial"}),
    "bug":      frozenset({"bug", "insect", "string", "signal", "sticky", "x-scissor", "megahorn"}),
    "rock":     frozenset({"rock", "stone", "boulder", "ancient", "meteor", "edge", "power-gem"}),
    "ghost":    frozenset({"ghost", "shadow", "phantom", "curse", "hex", "spite", "ominous"}),
    "dragon":   frozenset({"dragon", "draco", "outrage", "twister", "dual-wingbeat", "roar-of-time"}),
    "dark":     frozenset({"dark", "bite", "crunch", "sucker", "knock", "snarl", "brutal", "night", "feint"}),
    "steel":    frozenset({"iron", "steel", "metal", "flash-cannon", "meteor-mash", "bullet", "heavy-slam", "gyro"}),
    "fairy":    frozenset({"fairy", "moonblast", "dazzling", "play-rough", "gleam", "misty", "moonlight", "mystical"}),
    "normal":   frozenset({"body", "slam", "return", "frustration", "boomburst", "hyper-voice", "swift", "extreme"}),
}

_STRUGGLE_MOVE: dict = {
    "name": "Struggle",
    "api_name": "struggle",
    "type": "normal",
    "power": 50,
    "accuracy": None,
    "pp": 1,
    "damage_class": "physical",
    "target": "selected-pokemon",
    "stat_changes": [],
}


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _norm(move: dict) -> str:
    """Normalise a move dict to its api_name string for set lookups."""
    return (move.get("api_name") or move.get("name") or "").lower().replace(" ", "-").strip()


def _stab_coverage(pokemon_types: list[str]) -> frozenset[str]:
    """All types hit SE by the Pokémon's own STAB types."""
    result: set[str] = set()
    for t in pokemon_types:
        result |= SE_AGAINST.get(t, frozenset())
    return frozenset(result)


# ──────────────────────────────────────────────────────────────────────────
# Role classification
# ──────────────────────────────────────────────────────────────────────────

def classify_role(base_stats: dict) -> str:
    """
    Classify a Pokémon into one of six competitive roles.

    Returns one of:
        'physical_sweeper', 'special_sweeper', 'mixed_attacker',
        'physical_tank', 'special_tank', 'support'
    """
    atk    = base_stats.get("atk",    0)
    sp_atk = base_stats.get("sp_atk", 0)
    hp     = base_stats.get("hp",     0)
    defn   = base_stats.get("def",    0)
    sp_def = base_stats.get("sp_def", 0)
    speed  = base_stats.get("spd",    0)

    best_offence = max(atk, sp_atk)
    bulk = (hp + defn + sp_def) / 3.0

    # Mixed: both attack stats are close and both are high
    if abs(atk - sp_atk) <= 25 and min(atk, sp_atk) >= 85 and best_offence >= 95:
        return "mixed_attacker"

    # Physical sweeper
    if atk >= sp_atk + 15 and atk >= 85:
        return "physical_sweeper"

    # Special sweeper
    if sp_atk >= atk + 15 and sp_atk >= 85:
        return "special_sweeper"

    # Tank — high bulk, moderate offence
    if bulk >= 88 and best_offence < 90:
        return "physical_tank" if defn >= sp_def else "special_tank"

    # Support — low offence overall
    if best_offence < 75:
        return "support"

    # Default: whichever attack stat is higher
    return "physical_sweeper" if atk >= sp_atk else "special_sweeper"


# ──────────────────────────────────────────────────────────────────────────
# Move scoring  (the heart of the system)
# ──────────────────────────────────────────────────────────────────────────

def score_move(
    move: dict,
    pokemon_types: list[str],
    base_stats: dict,
    role: str,
    already_chosen_types: set[str],
) -> float:
    """
    Score a single move for this Pokémon.
    Higher = more valuable.  The score is additive across eight criteria:

    1.  Power base score         (0 – 15)
    2.  STAB bonus               (+20 if type matches)
    3.  New coverage bonus       (+4 per newly covered SE type, max +16)
    4.  Role-fit bonus           (0 – 15 for correct damage class / category)
    5.  Setup / utility bonus    (0 – 18 for setup, recovery, status, pivot)
    6.  Reliability bonus        (−0 to −5 for low accuracy)
    7.  Known competitive bonus  (+8 if in _HIGH_PRIORITY)
    8.  Type redundancy penalty  (−15 if same type already chosen)
    """
    api = _norm(move)
    mtype = (move.get("type") or "normal").lower()
    dc    = move.get("damage_class", "status")
    power = move.get("power") or 0
    acc   = move.get("accuracy") if move.get("accuracy") is not None else 100

    s = 0.0

    # ── 1. Power base ──────────────────────────────────────────────────────
    if power > 0:
        s += min(power / 10.0, 15.0)          # 80 pwr → 8, 150 pwr → 15

    # ── 2. STAB ────────────────────────────────────────────────────────────
    if mtype in pokemon_types:
        s += 20.0

    # ── 3. Type coverage ───────────────────────────────────────────────────
    stab_cov = _stab_coverage(pokemon_types)
    all_stab_types = set(pokemon_types) | stab_cov
    new_se = SE_AGAINST.get(mtype, frozenset()) - all_stab_types
    s += min(len(new_se) * 4.0, 16.0)

    # ── 4. Role fit ────────────────────────────────────────────────────────
    if role == "physical_sweeper":
        if dc == "physical" and power > 0:
            s += 12.0
        elif dc == "special" and power > 0:
            s -= 6.0
    elif role == "special_sweeper":
        if dc == "special" and power > 0:
            s += 12.0
        elif dc == "physical" and power > 0:
            s -= 6.0
    elif role == "mixed_attacker":
        if power > 0:
            s += 6.0                           # both classes equally useful
    elif role in ("physical_tank", "special_tank"):
        # Tanks care less about offence, more about bulk moves and utility
        if dc == "physical" and role == "physical_tank" and power > 0:
            s += 5.0
        elif dc == "special" and role == "special_tank" and power > 0:
            s += 5.0
    elif role == "support":
        # Support slightly prefers STAB for damage when it needs it
        if power > 0 and mtype in pokemon_types:
            s += 5.0

    # ── 5. Utility categories ──────────────────────────────────────────────
    if api in _SETUP:
        if role in ("physical_sweeper", "special_sweeper", "mixed_attacker"):
            s += 18.0
        else:
            s += 6.0

    if api in _RECOVERY:
        if role in ("physical_tank", "special_tank"):
            s += 18.0
        elif role == "support":
            s += 12.0
        else:
            s += 5.0

    if api in _STATUS:
        if role in ("physical_tank", "special_tank", "support"):
            s += 15.0
        else:
            s += 5.0

    if api in _PIVOT:
        s += 10.0                              # pivot is always useful

    # ── 6. Reliability ─────────────────────────────────────────────────────
    if power > 0:
        if acc < 100:
            s -= (100 - acc) / 25.0            # −1.2 per 30 % acc loss
        if acc < 80:
            s -= 2.0                           # extra penalty for unreliable

    # ── 7. Known competitive move ──────────────────────────────────────────
    if api in _HIGH_PRIORITY:
        s += 8.0

    # ── 8. Type redundancy ─────────────────────────────────────────────────
    if mtype in already_chosen_types and power > 0:
        s -= 15.0                              # strongly penalise same type

    # ── Edge cases ─────────────────────────────────────────────────────────
    # Very weak offensive moves are almost never worth a slot
    if 0 < power <= 40 and api not in _HIGH_PRIORITY:
        s -= 8.0

    return s


# ──────────────────────────────────────────────────────────────────────────
# Name-only pre-filter (no API calls required)
# ──────────────────────────────────────────────────────────────────────────

def prefilter_move_names(
    move_names: list[str],
    pokemon_types: list[str],
    limit: int = 22,
) -> list[str]:
    """
    Select up to *limit* move names from the full PokeAPI move list using
    only the move NAME as a signal.  No additional API calls are made.

    Priority order:
      1. Known high-priority competitive moves
      2. Names matching the Pokémon's own types (keyword heuristic)
      3. Known utility moves (setup / recovery / status / pivot)
      4. Everything else (capped to avoid excessive API calls)
    """
    type_kws: set[str] = set()
    for t in pokemon_types:
        type_kws |= _TYPE_KW.get(t, frozenset())

    scored: list[tuple[int, str]] = []
    for raw in move_names:
        name = raw.lower().strip()
        if name in _EXCLUDED:
            continue

        score = 0

        if name in _HIGH_PRIORITY:
            score += 50

        # Type keyword match (substring check for efficiency)
        if any(kw in name for kw in type_kws):
            score += 15

        if name in (_SETUP | _RECOVERY | _STATUS | _PIVOT):
            score += 20

        # Generic "good move" structural keywords
        if any(kw in name for kw in (
            "dance", "plot", "mind", "blast", "beam", "bolt", "punch",
            "fang", "claw", "edge", "cannon", "power", "storm", "wave",
        )):
            score += 6

        # Penalise obviously weak/trivial moves by name
        if any(kw in name for kw in (
            "scratch", "tackle", "pound", "peck", "splash",
            "whip", "growl", "leer", "tail-whip", "sand",
        )):
            score -= 8

        scored.append((score, name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:limit]]


# ──────────────────────────────────────────────────────────────────────────
# Core moveset builder
# ──────────────────────────────────────────────────────────────────────────

def _pick_iterative(
    pokemon_types: list[str],
    base_stats: dict,
    candidates: list[dict],
    role: str,
    max_same_type: int = 1,
) -> list[dict]:
    """
    Iteratively select 4 moves, rescoring after each pick to account for
    type redundancy.  Guarantees at least one STAB move whenever possible.
    """
    valid = [m for m in candidates if _norm(m) not in _EXCLUDED]
    if not valid:
        return [dict(_STRUGGLE_MOVE)] * 4

    chosen: list[dict] = []
    chosen_names: set[str] = set()
    chosen_types: set[str] = set()

    def _pick_best(pool: list[dict]) -> dict | None:
        best_s, best_m = -9999.0, None
        for m in pool:
            if _norm(m) in chosen_names:
                continue
            # Enforce max_same_type
            mtype = (m.get("type") or "normal").lower()
            if chosen_types.count_type(mtype) >= max_same_type:  # type: ignore[attr-defined]
                continue
            s = score_move(m, pokemon_types, base_stats, role, chosen_types)
            if s > best_s:
                best_s, best_m = s, m
        return best_m

    # We can't call .count_type on a set; track counts manually
    type_count: dict[str, int] = {}

    def pick_best_v2(pool: list[dict]) -> dict | None:
        best_s, best_m = -9999.0, None
        for m in pool:
            nm = _norm(m)
            if nm in chosen_names:
                continue
            mtype = (m.get("type") or "normal").lower()
            if type_count.get(mtype, 0) >= max_same_type and (m.get("power") or 0) > 0:
                continue
            s = score_move(m, pokemon_types, base_stats, role, chosen_types)
            if s > best_s:
                best_s, best_m = s, m
        return best_m

    # Pass 1 — guarantee at least 1 STAB move
    stab_pool = [
        m for m in valid
        if (m.get("type") or "").lower() in pokemon_types
        and (m.get("power") or 0) > 0
    ]
    if stab_pool:
        stab_pool.sort(
            key=lambda m: score_move(m, pokemon_types, base_stats, role, set()),
            reverse=True,
        )
        first = stab_pool[0]
        chosen.append(first)
        chosen_names.add(_norm(first))
        mtype = (first.get("type") or "normal").lower()
        chosen_types.add(mtype)
        type_count[mtype] = type_count.get(mtype, 0) + 1

    # Pass 2 — fill remaining 3 slots greedily
    for _ in range(4 - len(chosen)):
        pick = pick_best_v2(valid)
        if pick is None:
            break
        chosen.append(pick)
        chosen_names.add(_norm(pick))
        mtype = (pick.get("type") or "normal").lower()
        chosen_types.add(mtype)
        type_count[mtype] = type_count.get(mtype, 0) + 1

    # Pad with Struggle if fewer than 4 moves available
    while len(chosen) < 4:
        chosen.append(dict(_STRUGGLE_MOVE))

    return chosen[:4]


def build_moveset(
    pokemon_name: str,
    pokemon_types: list[str],
    base_stats: dict,
    candidate_moves: list[dict],
    mode: str = "competitive",
) -> list[dict]:
    """
    Select 4 moves for the given Pokémon from *candidate_moves*.

    Parameters
    ----------
    pokemon_name    : Display name (for logging only).
    pokemon_types   : List of type strings, e.g. ["fire", "flying"].
    base_stats      : Dict with keys hp/atk/def/sp_atk/sp_def/spd.
    candidate_moves : List of move dicts from PokeAPI (via get_move_data).
    mode            : One of "competitive" | "balanced" | "random".
                      "custom" is handled externally by the dashboard.

    Returns
    -------
    List of exactly 4 move dicts.
    """
    # Always filter hard-excluded moves first
    valid = [m for m in candidate_moves if _norm(m) not in _EXCLUDED and m]

    if mode == "random":
        result = valid[:4]
        while len(result) < 4:
            result.append(dict(_STRUGGLE_MOVE))
        return result

    if not valid:
        return [dict(_STRUGGLE_MOVE)] * 4

    role = classify_role(base_stats)
    max_same_type = 1 if mode == "competitive" else 2

    return _pick_iterative(pokemon_types, base_stats, valid, role, max_same_type)


# ──────────────────────────────────────────────────────────────────────────
# Utilities for UI
# ──────────────────────────────────────────────────────────────────────────

_ROLE_META: dict[str, dict] = {
    "physical_sweeper": {
        "label": "⚔️ Physical Sweeper",
        "color": "#e74c3c",
        "desc":  "High Attack. Prefers physical STAB + Swords Dance.",
    },
    "special_sweeper": {
        "label": "✨ Special Sweeper",
        "color": "#9b59b6",
        "desc":  "High Sp. Atk. Prefers special STAB + Nasty Plot.",
    },
    "mixed_attacker": {
        "label": "🔀 Mixed Attacker",
        "color": "#e67e22",
        "desc":  "Both Atk stats are high. Runs physical and special moves.",
    },
    "physical_tank": {
        "label": "🛡️ Physical Tank",
        "color": "#27ae60",
        "desc":  "High HP + Def. Prefers recovery and status infliction.",
    },
    "special_tank": {
        "label": "💎 Special Tank",
        "color": "#16a085",
        "desc":  "High HP + Sp. Def. Prefers Toxic and recovery.",
    },
    "support": {
        "label": "🌟 Support",
        "color": "#3498db",
        "desc":  "Low offence. Focuses on hazards, status, and pivoting.",
    },
}


def get_role_info(base_stats: dict) -> dict:
    """Return role classification and display metadata for a Pokémon."""
    role = classify_role(base_stats)
    return {"role": role, **_ROLE_META.get(role, {"label": role, "color": "#888", "desc": ""})}


def get_filtered_move_pool(
    candidate_moves: list[dict],
    pokemon_types: list[str],
    base_stats: dict,
    limit: int = 20,
) -> list[dict]:
    """
    Return the top-*limit* moves scored for this Pokémon, used to populate
    the 'Custom' mode editing dropdown in the dashboard.
    """
    role = classify_role(base_stats)
    valid = [m for m in candidate_moves if _norm(m) not in _EXCLUDED and m]
    scored = [
        (score_move(m, pokemon_types, base_stats, role, set()), m)
        for m in valid
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]
