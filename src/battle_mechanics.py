"""
battle_mechanics.py
~~~~~~~~~~~~~~~~~~~
Pure-function battle mechanics used by both the training env and the
live dashboard.  Nothing here touches Streamlit or RL-specific code.

Scope
-----
* Status conditions  (burn, poison, paralysis, sleep, freeze)
* Weather effects    (rain, sun, sandstorm, hail)
* Entry hazards      (stealth rock, spikes)
* MOVE_STATUS_TABLE  (move-name → (condition, chance_pct))

Design constraint: observation shape (28,) and action space (4) are never
touched here.  Status is already encoded in the obs as a binary flag.
"""

import random

# ---------------------------------------------------------------------------
# MOVE → STATUS TABLE
# Maps normalised move names (lowercase, hyphenated) to (condition, chance_%).
# Used by PokemonEnv._try_apply_status() and the dashboard's battle engine.
# ---------------------------------------------------------------------------

MOVE_STATUS_TABLE: dict[str, tuple[str, int]] = {
    # Burn
    "will-o-wisp":      ("burn", 100),
    "will-o-wisp-lite": ("burn", 100),
    "lava-plume":        ("burn", 30),
    "scald":             ("burn", 30),
    "flamethrower":      ("burn", 10),
    "fire-blast":        ("burn", 10),
    "flame-wheel":       ("burn", 10),
    "heat-wave":         ("burn", 10),
    "ember":             ("burn", 10),
    "sacred-fire":       ("burn", 50),
    "inferno":           ("burn", 100),
    # Paralysis
    "thunder-wave":  ("paralysis", 100),
    "glare":         ("paralysis", 100),
    "stun-spore":    ("paralysis", 75),
    "thunder":       ("paralysis", 30),
    "thunderbolt":   ("paralysis", 10),
    "discharge":     ("paralysis", 30),
    "spark":         ("paralysis", 30),
    "nuzzle":        ("paralysis", 100),
    "body-slam":     ("paralysis", 30),
    "lick":          ("paralysis", 30),
    # Poison
    "toxic":          ("poison", 100),
    "poison-powder":  ("poison", 75),
    "sludge-bomb":    ("poison", 30),
    "sludge":         ("poison", 30),
    "sludge-wave":    ("poison", 10),
    "cross-poison":   ("poison", 10),
    # Sleep
    "spore":         ("sleep", 100),
    "sleep-powder":  ("sleep", 75),
    "hypnosis":      ("sleep", 60),
    "dark-void":     ("sleep", 80),
    "lovely-kiss":   ("sleep", 75),
    "sing":          ("sleep", 55),
    "grass-whistle": ("sleep", 55),
    "yawn":          ("sleep", 100),
    # Freeze
    "blizzard":     ("freeze", 10),
    "ice-beam":     ("freeze", 10),
    "ice-punch":    ("freeze", 10),
    "powder-snow":  ("freeze", 10),
    "freeze-dry":   ("freeze", 10),
    "aurora-beam":  ("freeze", 10),
}

# ---------------------------------------------------------------------------
# WEATHER CONSTANTS
# ---------------------------------------------------------------------------

WEATHER_TURNS_DEFAULT = 5

WEATHER_DAMAGE_TYPES = {
    # Types that take 1.5× damage in their weather vs types immune to chip
    "sandstorm": {
        "chip_immune": {"rock", "ground", "steel"},  # no chip damage
        "chip_ratio": 1 / 16,
    },
    "hail": {
        "chip_immune": {"ice"},
        "chip_ratio": 1 / 16,
    },
}

WEATHER_MOVE_BOOST = {
    # (weather, move_type) → damage multiplier
    ("rain", "water"):   1.5,
    ("rain", "fire"):    0.5,
    ("sun", "fire"):     1.5,
    ("sun", "water"):    0.5,
}

# ---------------------------------------------------------------------------
# ENTRY HAZARD CONSTANTS
# ---------------------------------------------------------------------------

HAZARD_DAMAGE: dict[str, float] = {
    # damage ratios applied to current_hp when a Pokémon is sent in
    "stealth_rock": 1 / 8,   # ignores type for simplicity
    "spikes_1":     1 / 8,
    "spikes_2":     1 / 6,
    "spikes_3":     1 / 4,
}


# ---------------------------------------------------------------------------
# STATUS CONDITION FUNCTIONS
# ---------------------------------------------------------------------------

def try_apply_move_status(move: dict, target: dict) -> str:
    """
    Attempt to inflict a status condition from a move onto the target.

    Returns a human-readable log string if status was applied, else "".
    Does NOT apply if target already has a status condition.
    """
    if target.get("status") or target.get("debilitado"):
        return ""

    # Normalise the move name for table lookup
    raw_name = (move.get("api_name") or move.get("name") or "").lower()
    normalised = raw_name.strip().replace(" ", "-")

    entry = MOVE_STATUS_TABLE.get(normalised)
    if not entry:
        return ""

    condition, chance_pct = entry
    if random.randint(1, 100) > chance_pct:
        return ""

    target["status"] = condition
    if condition == "sleep":
        target["sleep_turns"] = random.randint(1, 3)

    _STATUS_PAST = {
        "burn":      "burned",
        "poison":    "poisoned",
        "paralysis": "paralyzed",
        "sleep":     "put to sleep",
        "freeze":    "frozen",
    }
    return f"{target['name']} was {_STATUS_PAST.get(condition, condition)}!"


def check_status_skip(pokemon: dict) -> tuple[bool, str]:
    """
    Check whether a Pokémon's status condition prevents it from moving.

    Returns (skip: bool, log_message: str).
    Modifies the pokemon dict in-place (e.g. decrements sleep_turns).
    """
    status = pokemon.get("status")
    name = pokemon.get("name", "?")

    if status == "sleep":
        turns_left = pokemon.get("sleep_turns", 1)
        if turns_left > 0:
            pokemon["sleep_turns"] = turns_left - 1
            return True, f"{name} is fast asleep!"
        # Wake up
        pokemon["status"] = None
        pokemon.pop("sleep_turns", None)
        return False, f"{name} woke up!"

    if status == "freeze":
        if random.random() < 0.20:   # 20 % thaw chance each turn
            pokemon["status"] = None
            return False, f"{name} thawed out!"
        return True, f"{name} is frozen solid!"

    if status == "paralysis":
        if random.random() < 0.25:   # 25 % full-paralysis chance
            return True, f"{name} is fully paralyzed! It can't move!"

    return False, ""


def get_status_chip_damage(pokemon: dict) -> tuple[float, str]:
    """
    Return (damage_ratio, log) for end-of-turn status chip damage.
    damage_ratio is expressed as a fraction of max HP (same scale as current_hp).
    Returns (0.0, "") if no chip damage applies.
    """
    status = pokemon.get("status")
    name = pokemon.get("name", "?")

    if status == "burn":
        chip = 1 / 16
        return chip, f"🔥 {name} is hurt by its burn! (−{chip * 100:.1f}%)"

    if status == "poison":
        chip = 1 / 8
        return chip, f"☠️ {name} is hurt by poison! (−{chip * 100:.1f}%)"

    return 0.0, ""


def get_paralysis_speed_factor(pokemon: dict) -> float:
    """Return the speed multiplier for a paralyzed Pokémon (0.5×), else 1.0."""
    return 0.5 if pokemon.get("status") == "paralysis" else 1.0


# ---------------------------------------------------------------------------
# WEATHER FUNCTIONS
# ---------------------------------------------------------------------------

def get_weather_damage_multiplier(move_type: str, weather: str | None) -> float:
    """Return the damage multiplier imposed by the current weather on a move type."""
    if not weather:
        return 1.0
    return WEATHER_MOVE_BOOST.get((weather, (move_type or "").lower()), 1.0)


def get_weather_chip_damage(pokemon: dict, weather: str | None) -> tuple[float, str]:
    """
    Return (chip, log) for end-of-turn weather chip damage.
    Applies to sandstorm and hail; Steel/Rock/Ground are immune to sandstorm,
    Ice is immune to hail.
    """
    if not weather or weather not in WEATHER_DAMAGE_TYPES:
        return 0.0, ""
    entry = WEATHER_DAMAGE_TYPES[weather]
    types = {t.lower() for t in (pokemon.get("types") or [])}
    if types & entry["chip_immune"]:
        return 0.0, ""
    chip = entry["chip_ratio"]
    name = pokemon.get("name", "?")
    icon = "🌪️" if weather == "sandstorm" else "🌨️"
    return chip, f"{icon} {name} is buffeted by the {weather}! (−{chip * 100:.1f}%)"


# ---------------------------------------------------------------------------
# ENTRY HAZARD FUNCTIONS
# ---------------------------------------------------------------------------

def get_hazard_entry_damage(pokemon: dict, hazards: set) -> tuple[float, str]:
    """
    Return (chip, log) when a Pokémon is sent onto a hazardous field.
    hazards: a set of strings, e.g. {"stealth_rock", "spikes_2"}.
    """
    chip = 0.0
    logs = []
    name = pokemon.get("name", "?")
    types = {t.lower() for t in (pokemon.get("types") or [])}

    if "stealth_rock" in hazards:
        dmg = HAZARD_DAMAGE["stealth_rock"]
        chip += dmg
        logs.append(f"🪨 {name} is hurt by Stealth Rock! (−{dmg * 100:.1f}%)")

    # Spikes (ground type is immune)
    if "ground" not in types:
        for tier in ("spikes_3", "spikes_2", "spikes_1"):
            if tier in hazards:
                dmg = HAZARD_DAMAGE[tier]
                chip += dmg
                logs.append(f"📌 {name} is hurt by Spikes! (−{dmg * 100:.1f}%)")
                break

    return chip, " | ".join(logs)
