"""
pokemon_forms.py
~~~~~~~~~~~~~~~~
Pure-Python form resolution for the Pokémon battle system.

No Streamlit, no API calls — all data is declarative.

Public API
----------
MEGA_STONE_MAP   : dict[str, dict]  — item name → form spec
GMAX_POKEMON     : dict[str, str]   — base Pokémon name → PokeAPI form name
resolve_form(base_name, item_name, flags) → dict

Why this module exists
----------------------
The previous approach detected Mega evolutions via ``if "ite" in item``, which
was unreliable (e.g. "leftovers" contains "te"), and mutated the Pokémon name
string in-place which broke sprite loading and made the system non-deterministic.

This module replaces that with:
- An explicit, exhaustive lookup table (no string inference)
- A single pure function that is always deterministic
- Clean separation from Streamlit / API code (importable in tests & training)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# MEGA STONE / FORM ITEM MAP
#
# Keys   : held-item canonical name, lowercase, as PokeAPI returns it
#           (hyphens between words, e.g. "charizardite-x")
#           Aliases for the space-separated variants are auto-generated below.
# Values : form spec dict
#   form_name  — PokeAPI pokemon-form name     (e.g. "charizard-mega-x")
#   form_type  — "mega"  (Primal Reversion also uses "mega" — same mechanic)
#   types      — override type list for the Mega form
#   stat_mult  — approximate multiplicative scaling of ALL base stats
#                (exact per-stat boosts vary; this is used for display only)
# ─────────────────────────────────────────────────────────────────────────────
MEGA_STONE_MAP: dict[str, dict] = {

    # ── Generation I ─────────────────────────────────────────────────────────
    "venusaurite": {
        "form_name": "venusaur-mega",
        "form_type": "mega",
        "types":     ["grass", "poison"],
        "stat_mult": 1.10,
    },
    "charizardite-x": {
        "form_name": "charizard-mega-x",
        "form_type": "mega",
        "types":     ["fire", "dragon"],
        "stat_mult": 1.10,
    },
    "charizardite-y": {
        "form_name": "charizard-mega-y",
        "form_type": "mega",
        "types":     ["fire", "flying"],
        "stat_mult": 1.10,
    },
    "blastoisinite": {
        "form_name": "blastoise-mega",
        "form_type": "mega",
        "types":     ["water"],
        "stat_mult": 1.10,
    },
    "beedrillite": {
        "form_name": "beedrill-mega",
        "form_type": "mega",
        "types":     ["bug", "poison"],
        "stat_mult": 1.10,
    },
    "pidgeotite": {
        "form_name": "pidgeot-mega",
        "form_type": "mega",
        "types":     ["normal", "flying"],
        "stat_mult": 1.10,
    },
    "alakazite": {
        "form_name": "alakazam-mega",
        "form_type": "mega",
        "types":     ["psychic"],
        "stat_mult": 1.10,
    },
    "slowbronite": {
        "form_name": "slowbro-mega",
        "form_type": "mega",
        "types":     ["water", "psychic"],
        "stat_mult": 1.10,
    },
    "gengarite": {
        "form_name": "gengar-mega",
        "form_type": "mega",
        "types":     ["ghost", "poison"],
        "stat_mult": 1.10,
    },
    "kangaskhanite": {
        "form_name": "kangaskhan-mega",
        "form_type": "mega",
        "types":     ["normal"],
        "stat_mult": 1.10,
    },
    "pinsirite": {
        "form_name": "pinsir-mega",
        "form_type": "mega",
        "types":     ["bug", "flying"],
        "stat_mult": 1.10,
    },
    "gyaradosite": {
        "form_name": "gyarados-mega",
        "form_type": "mega",
        "types":     ["water", "dark"],
        "stat_mult": 1.10,
    },
    "aerodactylite": {
        "form_name": "aerodactyl-mega",
        "form_type": "mega",
        "types":     ["rock", "flying"],
        "stat_mult": 1.10,
    },
    "mewtwonite-x": {
        "form_name": "mewtwo-mega-x",
        "form_type": "mega",
        "types":     ["psychic", "fighting"],
        "stat_mult": 1.12,
    },
    "mewtwonite-y": {
        "form_name": "mewtwo-mega-y",
        "form_type": "mega",
        "types":     ["psychic"],
        "stat_mult": 1.12,
    },

    # ── Generation II ────────────────────────────────────────────────────────
    "ampharosite": {
        "form_name": "ampharos-mega",
        "form_type": "mega",
        "types":     ["electric", "dragon"],
        "stat_mult": 1.10,
    },
    "steelixite": {
        "form_name": "steelix-mega",
        "form_type": "mega",
        "types":     ["steel", "ground"],
        "stat_mult": 1.10,
    },
    "scizorite": {
        "form_name": "scizor-mega",
        "form_type": "mega",
        "types":     ["bug", "steel"],
        "stat_mult": 1.10,
    },
    "heracronite": {
        "form_name": "heracross-mega",
        "form_type": "mega",
        "types":     ["bug", "fighting"],
        "stat_mult": 1.10,
    },
    "houndoomite": {
        "form_name": "houndoom-mega",
        "form_type": "mega",
        "types":     ["dark", "fire"],
        "stat_mult": 1.10,
    },
    "tyranitarite": {
        "form_name": "tyranitar-mega",
        "form_type": "mega",
        "types":     ["rock", "dark"],
        "stat_mult": 1.10,
    },

    # ── Generation III ───────────────────────────────────────────────────────
    "blazikenite": {
        "form_name": "blaziken-mega",
        "form_type": "mega",
        "types":     ["fire", "fighting"],
        "stat_mult": 1.10,
    },
    "swampertite": {
        "form_name": "swampert-mega",
        "form_type": "mega",
        "types":     ["water", "ground"],
        "stat_mult": 1.10,
    },
    "sceptilite": {
        "form_name": "sceptile-mega",
        "form_type": "mega",
        "types":     ["grass", "dragon"],
        "stat_mult": 1.10,
    },
    "gardevoirite": {
        "form_name": "gardevoir-mega",
        "form_type": "mega",
        "types":     ["psychic", "fairy"],
        "stat_mult": 1.10,
    },
    "mawilite": {
        "form_name": "mawile-mega",
        "form_type": "mega",
        "types":     ["steel", "fairy"],
        "stat_mult": 1.10,
    },
    "aggronite": {
        "form_name": "aggron-mega",
        "form_type": "mega",
        "types":     ["steel"],           # loses Rock type in Mega form
        "stat_mult": 1.10,
    },
    "medichamite": {
        "form_name": "medicham-mega",
        "form_type": "mega",
        "types":     ["fighting", "psychic"],
        "stat_mult": 1.10,
    },
    "manectite": {
        "form_name": "manectric-mega",
        "form_type": "mega",
        "types":     ["electric"],
        "stat_mult": 1.10,
    },
    "sharpedonite": {
        "form_name": "sharpedo-mega",
        "form_type": "mega",
        "types":     ["water", "dark"],
        "stat_mult": 1.10,
    },
    "cameruptite": {
        "form_name": "camerupt-mega",
        "form_type": "mega",
        "types":     ["fire", "ground"],
        "stat_mult": 1.10,
    },
    "altarianite": {
        "form_name": "altaria-mega",
        "form_type": "mega",
        "types":     ["dragon", "fairy"],
        "stat_mult": 1.10,
    },
    "banettite": {
        "form_name": "banette-mega",
        "form_type": "mega",
        "types":     ["ghost"],
        "stat_mult": 1.10,
    },
    "absolite": {
        "form_name": "absol-mega",
        "form_type": "mega",
        "types":     ["dark"],
        "stat_mult": 1.10,
    },
    "glalitite": {
        "form_name": "glalie-mega",
        "form_type": "mega",
        "types":     ["ice"],
        "stat_mult": 1.10,
    },
    "salamencite": {
        "form_name": "salamence-mega",
        "form_type": "mega",
        "types":     ["dragon", "flying"],
        "stat_mult": 1.10,
    },
    "metagrossite": {
        "form_name": "metagross-mega",
        "form_type": "mega",
        "types":     ["steel", "psychic"],
        "stat_mult": 1.10,
    },
    "latiasite": {
        "form_name": "latias-mega",
        "form_type": "mega",
        "types":     ["dragon", "psychic"],
        "stat_mult": 1.10,
    },
    "latiosite": {
        "form_name": "latios-mega",
        "form_type": "mega",
        "types":     ["dragon", "psychic"],
        "stat_mult": 1.10,
    },
    "rayquazite": {
        "form_name": "rayquaza-mega",
        "form_type": "mega",
        "types":     ["dragon", "flying"],
        "stat_mult": 1.12,
    },
    # Primal Reversion — triggered by the coloured Orbs, not Mega Stones,
    # but resolved identically as a form transformation.
    "red-orb": {
        "form_name": "groudon-primal",
        "form_type": "mega",
        "types":     ["ground", "fire"],
        "stat_mult": 1.12,
    },
    "blue-orb": {
        "form_name": "kyogre-primal",
        "form_type": "mega",
        "types":     ["water"],
        "stat_mult": 1.12,
    },

    # ── Generation IV ────────────────────────────────────────────────────────
    "lucarionite": {
        "form_name": "lucario-mega",
        "form_type": "mega",
        "types":     ["fighting", "steel"],
        "stat_mult": 1.10,
    },
    "lopunnite": {
        "form_name": "lopunny-mega",
        "form_type": "mega",
        "types":     ["normal", "fighting"],
        "stat_mult": 1.10,
    },
    "garchompite": {
        "form_name": "garchomp-mega",
        "form_type": "mega",
        "types":     ["dragon", "ground"],
        "stat_mult": 1.10,
    },
    "abomasite": {
        "form_name": "abomasnow-mega",
        "form_type": "mega",
        "types":     ["grass", "ice"],
        "stat_mult": 1.10,
    },
    "galladite": {
        "form_name": "gallade-mega",
        "form_type": "mega",
        "types":     ["psychic", "fighting"],
        "stat_mult": 1.10,
    },

    # ── Generation V ─────────────────────────────────────────────────────────
    "audinite": {
        "form_name": "audino-mega",
        "form_type": "mega",
        "types":     ["normal", "fairy"],
        "stat_mult": 1.10,
    },

    # ── Generation VI ────────────────────────────────────────────────────────
    "sablenite": {
        "form_name": "sableye-mega",
        "form_type": "mega",
        "types":     ["dark", "ghost"],
        "stat_mult": 1.10,
    },
    "diancite": {
        "form_name": "diancie-mega",
        "form_type": "mega",
        "types":     ["rock", "fairy"],
        "stat_mult": 1.12,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ALIAS TABLE  (auto-generated at import time, zero maintenance burden)
#
# Allows both hyphenated ("charizardite-x") and space-separated
# ("charizardite x") forms to resolve to the same canonical key.
# ─────────────────────────────────────────────────────────────────────────────
_MEGA_STONE_ALIASES: dict[str, str] = {}
for _canon_key in list(MEGA_STONE_MAP.keys()):
    _spaced     = _canon_key.replace("-", " ")
    _hyphenated = _canon_key.replace(" ", "-")
    if _spaced != _canon_key:
        _MEGA_STONE_ALIASES[_spaced]     = _canon_key
    if _hyphenated != _canon_key:
        _MEGA_STONE_ALIASES[_hyphenated] = _canon_key


# ─────────────────────────────────────────────────────────────────────────────
# GIGANTAMAX-CAPABLE POKÉMON
#
# key   : base Pokémon name, lowercase  (matches what PokeAPI returns)
# value : PokeAPI pokemon-form name for the G-Max form
# Source: https://bulbapedia.bulbagarden.net/wiki/Gigantamax
# ─────────────────────────────────────────────────────────────────────────────
GMAX_POKEMON: dict[str, str] = {
    # Kanto
    "venusaur":    "venusaur-gmax",
    "charizard":   "charizard-gmax",
    "blastoise":   "blastoise-gmax",
    "butterfree":  "butterfree-gmax",
    "pikachu":     "pikachu-gmax",
    "meowth":      "meowth-gmax",
    "machamp":     "machamp-gmax",
    "gengar":      "gengar-gmax",
    "kingler":     "kingler-gmax",
    "lapras":      "lapras-gmax",
    "eevee":       "eevee-gmax",
    "snorlax":     "snorlax-gmax",
    # Johto  (none with canonical G-Max forms in Sword/Shield)
    # Galar starters
    "rillaboom":   "rillaboom-gmax",
    "cinderace":   "cinderace-gmax",
    "inteleon":    "inteleon-gmax",
    # Galar regional
    "corviknight": "corviknight-gmax",
    "orbeetle":    "orbeetle-gmax",
    "drednaw":     "drednaw-gmax",
    "coalossal":   "coalossal-gmax",
    "flapple":     "flapple-gmax",
    "appletun":    "appletun-gmax",
    "sandaconda":  "sandaconda-gmax",
    "toxtricity":  "toxtricity-gmax",
    "centiskorch": "centiskorch-gmax",
    "hatterene":   "hatterene-gmax",
    "grimmsnarl":  "grimmsnarl-gmax",
    "alcremie":    "alcremie-gmax",
    "copperajah":  "copperajah-gmax",
    "duraludon":   "duraludon-gmax",
    "garbodor":    "garbodor-gmax",
    "melmetal":    "melmetal-gmax",
    # Urshifu — single-strike form is the base
    "urshifu":     "urshifu-single-strike-gmax",
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mega_key(item_key: str) -> str | None:
    """
    Return the canonical MEGA_STONE_MAP key for *item_key*, or None.

    Handles both hyphenated and space-separated item names transparently.
    """
    if item_key in MEGA_STONE_MAP:
        return item_key
    canon = _MEGA_STONE_ALIASES.get(item_key)
    if canon and canon in MEGA_STONE_MAP:
        return canon
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def resolve_form(
    base_name: str,
    item_name: str,
    flags: dict | None = None,
) -> dict:
    """
    Determine the game-form for a Pokémon given its held item and battle flags.

    This function is the SINGLE authoritative source for form resolution.
    It uses only explicit data tables — no string-inference heuristics.

    Parameters
    ----------
    base_name : str
        Canonical base-form name as returned by the PokeAPI, e.g. "charizard".
        Any capitalisation or leading/trailing whitespace is normalised.
    item_name : str
        Held item display or API name, e.g. "Charizardite X", "charizardite-x",
        or "Gigantamax Factor".  Normalised internally.
    flags : dict, optional
        Extra battle-state signals:
          "gmax"    : bool — Gigantamax-capable (item = "Gigantamax Factor")
          "dynamax" : bool — currently Dynamaxed (no form sprite, HP×2 only)

    Returns
    -------
    dict with keys:
        "form_name"  : str         — PokeAPI pokemon-name for this form,
                                     e.g. "charizard-mega-x"
        "form_type"  : str         — "mega" | "gmax" | "dynamax" | "base"
        "types"      : list | None — type override; None = keep base types
        "stat_mult"  : float       — multiplicative stat scaler (1.0 = no change)
        "hp_mult"    : float       — HP scaler (2.0 for Dynamax/G-Max, else 1.0)

    Examples
    --------
    >>> resolve_form("charizard", "Charizardite X")
    {"form_name": "charizard-mega-x", "form_type": "mega",
     "types": ["fire", "dragon"], "stat_mult": 1.10, "hp_mult": 1.0}

    >>> resolve_form("pikachu", "Gigantamax Factor")
    {"form_name": "pikachu-gmax", "form_type": "gmax",
     "types": None, "stat_mult": 1.0, "hp_mult": 2.0}

    >>> resolve_form("charizard", "Life Orb")
    {"form_name": "charizard", "form_type": "base",
     "types": None, "stat_mult": 1.0, "hp_mult": 1.0}
    """
    flags     = flags or {}
    norm_base = base_name.lower().strip().replace(" ", "-")
    item_key  = (item_name or "").lower().strip()

    # ── Priority 1 · Gigantamax ───────────────────────────────────────────────
    # Triggered by "Gigantamax Factor" item or explicit gmax flag.
    # Only Pokémon in GMAX_POKEMON can G-Max; others fall through.
    is_gmax = flags.get("gmax") or item_key in ("gigantamax factor", "gigantamax-factor")
    if is_gmax and norm_base in GMAX_POKEMON:
        return {
            "form_name": GMAX_POKEMON[norm_base],
            "form_type": "gmax",
            "types":     None,
            "stat_mult": 1.0,
            "hp_mult":   2.0,
        }

    # ── Priority 2 · Mega Evolution (explicit table lookup ONLY) ─────────────
    # NO string-inference heuristics — every form is declared explicitly.
    mega_key = _resolve_mega_key(item_key)
    if mega_key:
        spec = MEGA_STONE_MAP[mega_key]
        return {
            "form_name": spec["form_name"],
            "form_type": spec["form_type"],   # "mega" (includes Primal)
            "types":     spec.get("types"),
            "stat_mult": spec.get("stat_mult", 1.0),
            "hp_mult":   1.0,
        }

    # ── Priority 3 · Dynamax ─────────────────────────────────────────────────
    # No visual form change; HP is doubled for display.
    is_dynamax = flags.get("dynamax") or item_key in ("dynamax band", "dynamax-band")
    if is_dynamax:
        return {
            "form_name": norm_base,
            "form_type": "dynamax",
            "types":     None,
            "stat_mult": 1.0,
            "hp_mult":   2.0,
        }

    # ── Priority 4 · Base form ───────────────────────────────────────────────
    return {
        "form_name": norm_base,
        "form_type": "base",
        "types":     None,
        "stat_mult": 1.0,
        "hp_mult":   1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASE POKÉMON VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
# Why this matters
# ----------------
# PokeAPI's /pokemon endpoint returns ALL entries — base species AND every form
# variant (e.g. "charizard-mega-x", "raichu-alola", "pikachu-gmax").  If these
# reach the team-builder UI they break sprite loading, battle initialisation,
# and AI observations, because the battle engine expects canonical base-species
# names.  Forms must only exist as the OUTPUT of resolve_form(), never as
# selectable inputs.
#
# Primary defence  : use the /pokemon-species endpoint (base species only).
# Secondary defence: is_base_pokemon() as a belt-and-suspenders filter.
# ─────────────────────────────────────────────────────────────────────────────

# All known non-base form suffixes used by PokeAPI names.
# Ordered from longest to shortest so normalize_pokemon_name() can strip greedily.
INVALID_FORM_SUFFIXES: tuple[str, ...] = (
    # Mega / Primal
    "-mega-x", "-mega-y", "-mega",
    "-primal",
    # Gigantamax
    "-gmax",
    # Regional forms
    "-alolan", "-alola",
    "-galarian", "-galar",
    "-hisuian", "-hisui",
    "-paldean", "-paldea",
    # Totem forms
    "-totem",
    # Legendary / mythical alternate forms
    "-origin",          # Giratina, Dialga, Palkia
    "-altered",         # Giratina
    "-black", "-white", # Kyurem
    "-resolute",        # Keldeo
    "-therian",         # Forces of Nature
    "-pirouette",       # Meloetta
    "-aria",            # Meloetta
    "-blade",           # Aegislash
    "-sky",             # Shaymin
    "-crowned",         # Zacian / Zamazenta
    "-eternamax",       # Eternatus
    "-single-strike", "-rapid-strike",  # Urshifu
    "-50", "-10", "-complete",           # Zygarde
    # Rotom appliances
    "-heat", "-wash", "-frost", "-fan", "-mow",
    # Deoxys forms
    "-attack", "-defense", "-speed",
    # Darmanitan
    "-zen", "-galarian-zen",
    # Lycanroc
    "-dusk", "-midnight",
    # Wishiwashi
    "-school",
    # Greninja
    "-ash", "-battle-bond",
    # Pikachu costume forms
    "-original-cap", "-hoenn-cap", "-sinnoh-cap", "-unova-cap",
    "-kalos-cap", "-alola-cap", "-partner-cap", "-world-cap",
    "-cosplay",
    # Castform
    "-sunny", "-rainy", "-snowy",
    # Shellos / Gastrodon
    "-east", "-west",
    # Paldean Tauros
    "-combat", "-blaze", "-aqua",
    # Basculin
    "-blue-striped", "-white-striped", "-striped",
    # Toxtricity
    "-low-key", "-amped",
    # Morpeko
    "-hangry",
    # Eiscue
    "-noice-face",
    # Dudunsparce
    "-three-segment",
    # Ursaluna
    "-bloodmoon",
    # Urshifu Gmax (handled above but listed for clarity)
    # Indeedee / Meowstic
    "-male", "-female",
    # Wormadam
    "-plant", "-sandy", "-trash",
    # Minior
    "-meteor",
    # Mimikyu
    "-busted",
    # Oricorio
    "-pom-pom", "-pau", "-sensu",
    # Silvally / Arceus types are NOT suffixes (they're different entries)
    # Kyurem fused forms
    "-white", "-black",
)


def is_base_pokemon(name: str) -> bool:
    """
    Return True if *name* refers to a BASE Pokémon species, not a form variant.

    Handles both PokeAPI lowercase-hyphenated names ("charizard-mega-x") and
    UI display names with spaces/title-case ("Charizard Mega X").

    Parameters
    ----------
    name : str
        Any representation of the Pokémon name.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_base_pokemon("charizard")          # True
    >>> is_base_pokemon("Charizard")          # True
    >>> is_base_pokemon("charizard-mega-x")   # False
    >>> is_base_pokemon("Charizard Mega X")   # False
    >>> is_base_pokemon("raichu-alola")       # False
    >>> is_base_pokemon("mr-mime")            # True  (hyphen in base name)
    >>> is_base_pokemon("tapu-koko")          # True  (hyphen in base name)
    """
    normalised = name.lower().strip().replace(" ", "-")
    return not any(normalised.endswith(suffix) for suffix in INVALID_FORM_SUFFIXES)


def normalize_pokemon_name(name: str) -> str:
    """
    Strip a known form suffix from a Pokémon name, returning the base species.

    This is the CORRECT normaliser — it does NOT use ``name.split("-")[0]``
    which would corrupt multi-word Pokémon like "mr-mime" → "mr" or
    "tapu-koko" → "tapu".  Instead it checks against the explicit suffix list.

    Parameters
    ----------
    name : str
        Pokémon name, possibly including a form suffix.

    Returns
    -------
    str
        Lowercase, hyphenated base species name.

    Examples
    --------
    >>> normalize_pokemon_name("charizard-mega-x")   # "charizard"
    >>> normalize_pokemon_name("raichu-alola")        # "raichu"
    >>> normalize_pokemon_name("mr-mime")             # "mr-mime"   (unchanged)
    >>> normalize_pokemon_name("tapu-koko")           # "tapu-koko" (unchanged)
    """
    normalised = name.lower().strip().replace(" ", "-")
    # Suffixes are ordered longest-first; strip the first match found.
    for suffix in INVALID_FORM_SUFFIXES:
        if normalised.endswith(suffix):
            return normalised[: -len(suffix)]
    return normalised


def normalize_showdown_name(name: str) -> str:
    """
    Convert any Pokémon name string to the URL slug used by the Pokémon
    Showdown CDN.

    The Showdown slug format is strictly ``[a-z0-9-]`` — lowercase letters,
    digits and hyphens only, with no dots, apostrophes, colons, spaces or
    other punctuation.  The rules below are derived from Showdown's own
    client-side normalization logic.

    Rules applied in order
    ──────────────────────
    1. Strip surrounding whitespace and lowercase.
    2. Apply explicit special-case overrides (Nidoran♀/♂, Type: Null …).
    3. Remove dots, apostrophes, colons and similar punctuation.
    4. Replace every run of whitespace with a single hyphen.
    5. Strip every character that is not ``[a-z0-9-]``.
    6. Collapse consecutive hyphens into one.
    7. Strip leading/trailing hyphens.

    Parameters
    ----------
    name : str
        Any human-readable Pokémon name, with or without a form suffix.
        Examples: "Mr. Mime", "Mr Mime", "Kommo-o", "Kommo O",
                  "Farfetch'd", "Type: Null", "Nidoran♀".

    Returns
    -------
    str
        Showdown-compatible slug, never empty (falls back to "unknown").

    Examples
    --------
    >>> normalize_showdown_name("Mr. Mime")    # "mr-mime"
    >>> normalize_showdown_name("Mr Mime")     # "mr-mime"
    >>> normalize_showdown_name("Mime Jr.")    # "mime-jr"
    >>> normalize_showdown_name("Kommo-o")     # "kommo-o"
    >>> normalize_showdown_name("Kommo O")     # "kommo-o"
    >>> normalize_showdown_name("Type: Null")  # "type-null"
    >>> normalize_showdown_name("Tapu Koko")   # "tapu-koko"
    >>> normalize_showdown_name("Farfetch'd")  # "farfetchd"
    >>> normalize_showdown_name("Nidoran♀")   # "nidoran-f"
    >>> normalize_showdown_name("Nidoran♂")   # "nidoran-m"
    """
    import re as _re

    # ── Special cases that cannot be handled by generic rules ─────────────────
    # Nidoran gender symbols have no ASCII representation in Showdown slugs.
    # Type: Null has a colon which the generic rule would remove correctly,
    # but the explicit entry here makes intent crystal-clear.
    _SPECIAL: dict[str, str] = {
        "nidoran♀":   "nidoran-f",
        "nidoran-f":  "nidoran-f",
        "nidoran♂":   "nidoran-m",
        "nidoran-m":  "nidoran-m",
        "type: null": "type-null",
        "type:null":  "type-null",
        "type - null":"type-null",
        "jangmo-o":   "jangmo-o",
        "hakamo-o":   "hakamo-o",
        "kommo-o":    "kommo-o",
        "ho-oh":      "ho-oh",
        "porygon-z":  "porygon-z",
        "mr. mime":   "mr-mime",
        "mr mime":    "mr-mime",
        "mime jr.":   "mime-jr",
        "mime jr":    "mime-jr",
    }

    key = name.strip().lower()
    if key in _SPECIAL:
        return _SPECIAL[key]

    s = key
    # Rule 3 — remove punctuation that must not become hyphens
    s = s.replace(".", "").replace("'", "").replace("\u2019", "").replace(":", "")
    # Rule 4 — whitespace → hyphen
    s = _re.sub(r"\s+", "-", s)
    # Rule 5 — strip everything outside [a-z0-9-]
    s = _re.sub(r"[^a-z0-9\-]", "", s)
    # Rule 6 — collapse double (or more) hyphens
    s = _re.sub(r"-{2,}", "-", s)
    # Rule 7 — strip edge hyphens
    s = s.strip("-")

    return s or "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# REGIONAL FORMS
#
# Explicit mapping of every regional-form slug (PokeAPI / Showdown convention)
# to its type override and originating region.
#
# KEY DESIGN DECISIONS
# ─────────────────────
# • Keys are lowercase hyphenated Showdown slugs — same convention as the
#   rest of this module.  Both "-alola" and "-alolan" aliases are listed so
#   either source (PokeAPI vs Showdown) works transparently.
# • "types" overrides only where the regional form differs from the base.
#   None = inherit base types (the regional form keeps the original typing).
# • "stat_mult": 1.0 — regional forms keep base stats by default.  Override
#   only where there is a meaningful stat difference (e.g. Alolan Geodude line
#   keeps the same total but redistributed — modelled as 1.0 for safety).
# • These entries are treated as SEPARATE species by the battle engine: they
#   are never produced by resolve_form() (which handles mid-battle transforms),
#   but are valid team-slot entries selected at team-creation time.
# ─────────────────────────────────────────────────────────────────────────────

REGIONAL_FORMS: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # ALOLAN FORMS  (Sun & Moon / USUM)
    # ══════════════════════════════════════════════════════════════════════════
    "rattata-alola":      {"region": "alola", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "raticate-alola":     {"region": "alola", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "raichu-alola":       {"region": "alola", "types": ["electric", "psychic"],  "stat_mult": 1.0},
    "sandshrew-alola":    {"region": "alola", "types": ["ice", "steel"],         "stat_mult": 1.0},
    "sandslash-alola":    {"region": "alola", "types": ["ice", "steel"],         "stat_mult": 1.0},
    "vulpix-alola":       {"region": "alola", "types": ["ice"],                  "stat_mult": 1.0},
    "ninetales-alola":    {"region": "alola", "types": ["ice", "fairy"],         "stat_mult": 1.0},
    "diglett-alola":      {"region": "alola", "types": ["ground", "steel"],      "stat_mult": 1.0},
    "dugtrio-alola":      {"region": "alola", "types": ["ground", "steel"],      "stat_mult": 1.0},
    "meowth-alola":       {"region": "alola", "types": ["dark"],                 "stat_mult": 1.0},
    "persian-alola":      {"region": "alola", "types": ["dark"],                 "stat_mult": 1.0},
    "geodude-alola":      {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "graveler-alola":     {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "golem-alola":        {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "grimer-alola":       {"region": "alola", "types": ["poison", "dark"],       "stat_mult": 1.0},
    "muk-alola":          {"region": "alola", "types": ["poison", "dark"],       "stat_mult": 1.0},
    "exeggutor-alola":    {"region": "alola", "types": ["grass", "dragon"],      "stat_mult": 1.0},
    "marowak-alola":      {"region": "alola", "types": ["fire", "ghost"],        "stat_mult": 1.0},
    # ── Alolan aliases (PokeAPI uses "-alolan" suffix) ────────────────────────
    "rattata-alolan":     {"region": "alola", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "raticate-alolan":    {"region": "alola", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "raichu-alolan":      {"region": "alola", "types": ["electric", "psychic"],  "stat_mult": 1.0},
    "sandshrew-alolan":   {"region": "alola", "types": ["ice", "steel"],         "stat_mult": 1.0},
    "sandslash-alolan":   {"region": "alola", "types": ["ice", "steel"],         "stat_mult": 1.0},
    "vulpix-alolan":      {"region": "alola", "types": ["ice"],                  "stat_mult": 1.0},
    "ninetales-alolan":   {"region": "alola", "types": ["ice", "fairy"],         "stat_mult": 1.0},
    "diglett-alolan":     {"region": "alola", "types": ["ground", "steel"],      "stat_mult": 1.0},
    "dugtrio-alolan":     {"region": "alola", "types": ["ground", "steel"],      "stat_mult": 1.0},
    "meowth-alolan":      {"region": "alola", "types": ["dark"],                 "stat_mult": 1.0},
    "persian-alolan":     {"region": "alola", "types": ["dark"],                 "stat_mult": 1.0},
    "geodude-alolan":     {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "graveler-alolan":    {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "golem-alolan":       {"region": "alola", "types": ["rock", "electric"],     "stat_mult": 1.0},
    "grimer-alolan":      {"region": "alola", "types": ["poison", "dark"],       "stat_mult": 1.0},
    "muk-alolan":         {"region": "alola", "types": ["poison", "dark"],       "stat_mult": 1.0},
    "exeggutor-alolan":   {"region": "alola", "types": ["grass", "dragon"],      "stat_mult": 1.0},
    "marowak-alolan":     {"region": "alola", "types": ["fire", "ghost"],        "stat_mult": 1.0},

    # ══════════════════════════════════════════════════════════════════════════
    # GALARIAN FORMS  (Sword & Shield)
    # ══════════════════════════════════════════════════════════════════════════
    "meowth-galar":       {"region": "galar", "types": ["steel"],                "stat_mult": 1.0},
    "ponyta-galar":       {"region": "galar", "types": ["psychic"],              "stat_mult": 1.0},
    "rapidash-galar":     {"region": "galar", "types": ["psychic", "fairy"],     "stat_mult": 1.0},
    "slowpoke-galar":     {"region": "galar", "types": ["psychic"],              "stat_mult": 1.0},
    "slowbro-galar":      {"region": "galar", "types": ["poison", "psychic"],    "stat_mult": 1.0},
    "slowking-galar":     {"region": "galar", "types": ["poison", "psychic"],    "stat_mult": 1.0},
    "farfetchd-galar":    {"region": "galar", "types": ["fighting"],             "stat_mult": 1.0},
    "weezing-galar":      {"region": "galar", "types": ["poison", "fairy"],      "stat_mult": 1.0},
    "mr-mime-galar":      {"region": "galar", "types": ["ice", "psychic"],       "stat_mult": 1.0},
    "articuno-galar":     {"region": "galar", "types": ["psychic", "flying"],    "stat_mult": 1.0},
    "zapdos-galar":       {"region": "galar", "types": ["fighting", "flying"],   "stat_mult": 1.0},
    "moltres-galar":      {"region": "galar", "types": ["dark", "flying"],       "stat_mult": 1.0},
    "slowking-galar":     {"region": "galar", "types": ["poison", "psychic"],    "stat_mult": 1.0},
    "corsola-galar":      {"region": "galar", "types": ["ghost"],                "stat_mult": 1.0},
    "zigzagoon-galar":    {"region": "galar", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "linoone-galar":      {"region": "galar", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "darumaka-galar":     {"region": "galar", "types": ["ice"],                  "stat_mult": 1.0},
    "darmanitan-galar":   {"region": "galar", "types": ["ice"],                  "stat_mult": 1.0},
    "yamask-galar":       {"region": "galar", "types": ["ground", "ghost"],      "stat_mult": 1.0},
    "stunfisk-galar":     {"region": "galar", "types": ["ground", "steel"],      "stat_mult": 1.0},
    # Galarian aliases
    "meowth-galarian":    {"region": "galar", "types": ["steel"],                "stat_mult": 1.0},
    "ponyta-galarian":    {"region": "galar", "types": ["psychic"],              "stat_mult": 1.0},
    "rapidash-galarian":  {"region": "galar", "types": ["psychic", "fairy"],     "stat_mult": 1.0},
    "slowpoke-galarian":  {"region": "galar", "types": ["psychic"],              "stat_mult": 1.0},
    "slowbro-galarian":   {"region": "galar", "types": ["poison", "psychic"],    "stat_mult": 1.0},
    "corsola-galarian":   {"region": "galar", "types": ["ghost"],                "stat_mult": 1.0},
    "zigzagoon-galarian": {"region": "galar", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "linoone-galarian":   {"region": "galar", "types": ["dark", "normal"],       "stat_mult": 1.0},
    "weezing-galarian":   {"region": "galar", "types": ["poison", "fairy"],      "stat_mult": 1.0},
    "stunfisk-galarian":  {"region": "galar", "types": ["ground", "steel"],      "stat_mult": 1.0},

    # ══════════════════════════════════════════════════════════════════════════
    # HISUIAN FORMS  (Legends: Arceus)
    # ══════════════════════════════════════════════════════════════════════════
    "growlithe-hisui":    {"region": "hisui", "types": ["fire", "rock"],         "stat_mult": 1.0},
    "arcanine-hisui":     {"region": "hisui", "types": ["fire", "rock"],         "stat_mult": 1.0},
    "voltorb-hisui":      {"region": "hisui", "types": ["electric", "grass"],    "stat_mult": 1.0},
    "electrode-hisui":    {"region": "hisui", "types": ["electric", "grass"],    "stat_mult": 1.0},
    "typhlosion-hisui":   {"region": "hisui", "types": ["fire", "ghost"],        "stat_mult": 1.0},
    "qwilfish-hisui":     {"region": "hisui", "types": ["dark", "poison"],       "stat_mult": 1.0},
    "sneasel-hisui":      {"region": "hisui", "types": ["fighting", "poison"],   "stat_mult": 1.0},
    "samurott-hisui":     {"region": "hisui", "types": ["water", "dark"],        "stat_mult": 1.0},
    "lilligant-hisui":    {"region": "hisui", "types": ["grass", "fighting"],    "stat_mult": 1.0},
    "basculin-hisui":     {"region": "hisui", "types": ["water"],                "stat_mult": 1.0},
    "zorua-hisui":        {"region": "hisui", "types": ["normal", "ghost"],      "stat_mult": 1.0},
    "zoroark-hisui":      {"region": "hisui", "types": ["normal", "ghost"],      "stat_mult": 1.0},
    "braviary-hisui":     {"region": "hisui", "types": ["psychic", "flying"],    "stat_mult": 1.0},
    "sliggoo-hisui":      {"region": "hisui", "types": ["steel", "dragon"],      "stat_mult": 1.0},
    "goodra-hisui":       {"region": "hisui", "types": ["steel", "dragon"],      "stat_mult": 1.0},
    "avalugg-hisui":      {"region": "hisui", "types": ["ice", "rock"],          "stat_mult": 1.0},
    "decidueye-hisui":    {"region": "hisui", "types": ["grass", "fighting"],    "stat_mult": 1.0},
    "kleavor":            {"region": "hisui", "types": ["bug", "rock"],          "stat_mult": 1.0},
    "ursaluna":           {"region": "hisui", "types": ["ground", "normal"],     "stat_mult": 1.0},
    "basculegion":        {"region": "hisui", "types": ["water", "ghost"],       "stat_mult": 1.0},
    "sneasler":           {"region": "hisui", "types": ["fighting", "poison"],   "stat_mult": 1.0},
    "overqwil":           {"region": "hisui", "types": ["dark", "poison"],       "stat_mult": 1.0},
    "enamorus":           {"region": "hisui", "types": ["fairy", "flying"],      "stat_mult": 1.0},
    # Hisuian aliases
    "growlithe-hisuian":  {"region": "hisui", "types": ["fire", "rock"],         "stat_mult": 1.0},
    "arcanine-hisuian":   {"region": "hisui", "types": ["fire", "rock"],         "stat_mult": 1.0},
    "voltorb-hisuian":    {"region": "hisui", "types": ["electric", "grass"],    "stat_mult": 1.0},
    "electrode-hisuian":  {"region": "hisui", "types": ["electric", "grass"],    "stat_mult": 1.0},
    "zoroark-hisuian":    {"region": "hisui", "types": ["normal", "ghost"],      "stat_mult": 1.0},
    "braviary-hisuian":   {"region": "hisui", "types": ["psychic", "flying"],    "stat_mult": 1.0},
    "goodra-hisuian":     {"region": "hisui", "types": ["steel", "dragon"],      "stat_mult": 1.0},
    "decidueye-hisuian":  {"region": "hisui", "types": ["grass", "fighting"],    "stat_mult": 1.0},

    # ══════════════════════════════════════════════════════════════════════════
    # PALDEAN FORMS  (Scarlet & Violet)
    # ══════════════════════════════════════════════════════════════════════════
    "tauros-paldea":          {"region": "paldea", "types": ["fighting"],            "stat_mult": 1.0},
    "tauros-paldea-combat":   {"region": "paldea", "types": ["fighting"],            "stat_mult": 1.0},
    "tauros-paldea-blaze":    {"region": "paldea", "types": ["fighting", "fire"],    "stat_mult": 1.0},
    "tauros-paldea-aqua":     {"region": "paldea", "types": ["fighting", "water"],   "stat_mult": 1.0},
    "wooper-paldea":          {"region": "paldea", "types": ["poison", "ground"],    "stat_mult": 1.0},
    # Paldean aliases
    "tauros-paldean":         {"region": "paldea", "types": ["fighting"],            "stat_mult": 1.0},
    "wooper-paldean":         {"region": "paldea", "types": ["poison", "ground"],    "stat_mult": 1.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# SHINY SYSTEM
#
# Shiny Pokémon are purely cosmetic variants (different sprite, same stats).
# The system is:
#   1. DETERMINISTIC — shiny status is rolled once at Pokémon creation time
#      using the environment's seeded RNG.  Same seed → same shiny outcome.
#   2. CONFIGURABLE — SHINY_RATE is a module-level constant; pass a custom
#      rate to roll_shiny() for testing or special modes.
#   3. RL-SAFE — no randomness outside the seeded generator.  Shiny does NOT
#      affect stats, so observations are unchanged.  The "shiny" field is
#      stored in Pokémon state purely for visual rendering by the UI.
#
# Default rate: 1 / 4096 (modern games, post-Gen VI standard).
# Test mode:    pass rate=1.0 to roll_shiny() for guaranteed shiny in tests.
# Disabled:     pass rate=0.0 to roll_shiny() to guarantee no shiny.
# ─────────────────────────────────────────────────────────────────────────────

SHINY_RATE: float = 1.0 / 4096.0
"""
Default probability of a Pokémon being shiny (1 in 4096).

Matches the modern game mechanic introduced in Generation VI.
Pass a custom rate to roll_shiny() to override per-experiment.
"""


def roll_shiny(rng: object, rate: float = SHINY_RATE) -> bool:
    """
    Roll whether a Pokémon is shiny using the caller's seeded RNG.

    DESIGN CONTRACT
    ───────────────
    • Deterministic: identical seed → identical shiny outcome.
    • No hidden state: all randomness flows through the caller's ``rng``.
    • Caller owns the RNG: ``rng`` must be a ``numpy.random.Generator``
      (e.g. ``np.random.default_rng(seed)`` or ``env.np_random``).

    Parameters
    ----------
    rng : numpy.random.Generator
        The caller's seeded RNG.  Must expose a ``random()`` method that
        returns a float in [0, 1) — compatible with both ``np.random.default_rng``
        and Gymnasium's ``np_random``.
    rate : float, optional
        Probability of shiny (default ``SHINY_RATE = 1/4096``).
        Pass 0.0 for never-shiny, 1.0 for always-shiny (useful in tests).

    Returns
    -------
    bool
        True if the roll succeeds (Pokémon is shiny), False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> roll_shiny(rng)                   # normal rate, almost always False
    False
    >>> roll_shiny(rng, rate=1.0)         # guaranteed shiny
    True
    >>> roll_shiny(rng, rate=0.0)         # guaranteed normal
    False
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return bool(float(rng.random()) < rate)


def is_regional_form(name: str) -> bool:
    """
    Return True if *name* refers to a known regional form variant.

    Parameters
    ----------
    name : str
        Any Pokémon name representation.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_regional_form("raichu-alola")    # True
    >>> is_regional_form("raichu")          # False
    >>> is_regional_form("Galarian Ponyta") # True
    """
    slug = name.lower().strip().replace(" ", "-")
    if slug in REGIONAL_FORMS:
        return True
    # Also match display-name forms like "Alolan Raichu", "Galarian Ponyta"
    _REGION_PREFIXES = ("alolan-", "galarian-", "hisuian-", "paldean-")
    if any(slug.startswith(p) for p in _REGION_PREFIXES):
        # Reorder to suffix convention: "alolan-raichu" → "raichu-alola"
        for prefix, suffix in [
            ("alolan-", "-alola"), ("galarian-", "-galar"),
            ("hisuian-", "-hisui"), ("paldean-", "-paldea"),
        ]:
            if slug.startswith(prefix):
                base = slug[len(prefix):]
                return f"{base}{suffix}" in REGIONAL_FORMS
    return False


def get_regional_form_info(name: str) -> dict | None:
    """
    Return the REGIONAL_FORMS entry for *name*, or None if not a regional form.

    Handles both suffix-style ("raichu-alola") and prefix-style ("Alolan Raichu").

    Parameters
    ----------
    name : str
        Any Pokémon name.

    Returns
    -------
    dict | None
        REGIONAL_FORMS entry with keys "region", "types", "stat_mult",
        or None if not a recognised regional form.
    """
    slug = name.lower().strip().replace(" ", "-")
    # Direct suffix-style lookup
    if slug in REGIONAL_FORMS:
        return REGIONAL_FORMS[slug]
    # Prefix-style: "alolan-raichu" → "raichu-alola"
    for prefix, suffix in [
        ("alolan-", "-alola"), ("galarian-", "-galar"),
        ("hisuian-", "-hisui"), ("paldean-", "-paldea"),
    ]:
        if slug.startswith(prefix):
            base = slug[len(prefix):]
            candidate = f"{base}{suffix}"
            if candidate in REGIONAL_FORMS:
                return REGIONAL_FORMS[candidate]
    return None
