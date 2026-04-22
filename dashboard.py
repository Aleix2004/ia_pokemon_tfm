import base64
import json as _json
import logging
import os
import random
import re as _re
import sqlite3
import time

import pandas as pd
import requests
import streamlit as st

_log = logging.getLogger(__name__)

from src.battle_utils import (
    apply_stat_stages,
    build_type_chart_rows,
    describe_effectiveness,
    format_name,
    get_type_multiplier,
)
# ── Layer boundary: the dashboard uses the Game Engine layer, NOT the ──────
# ── training environment.  PokemonEnv is never imported here.          ──────
from src.game_engine.battle_engine import BattleEngine
from src.model_compat import check_model_compatibility, require_compatible_model
from src.type_colors import (
    get_type_colors,
    get_type_emoji,
    hp_bar_color,
    status_badge_html,
    type_badge_html,
    weather_badge_html,
)
from src.ai_advisor import get_greedy_action, get_hybrid_action, get_ia_switch_decision
from src.pokemon_forms import GMAX_POKEMON, MEGA_STONE_MAP, is_base_pokemon, normalize_pokemon_name, resolve_form
from src.sprite_registry import (
    DEFAULT_INDEX_PATH,
    FALLBACK_PATH,
    RegistryMeta,
    SpritePaths,
    get_sprite,
    init_sprite_registry,
    is_initialized,
    normalize_showdown_name,
)
from src.competitive_movesets import (
    build_moveset,
    get_filtered_move_pool,
    get_role_info,
    prefilter_move_names,
)


st.set_page_config(layout="wide", page_title="Pokemon AI TFM Dashboard", page_icon="🧪")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SPRITE BOOTSTRAP — runs ONCE at module load, before ANY get_sprite() call
# ─────────────────────────────────────────────────────────────────────────────
#
# PURPOSE
# ───────
# Guarantee that the minimum sprite asset infrastructure exists so that
# init_sprite_registry() ALWAYS succeeds, regardless of whether the download
# script has been run.
#
# The three-layer defence:
#   Layer 1 (this function):  sprite_index.json always exists before init
#   Layer 2 (_init_sprites):  @st.cache_resource — only one init per process
#   Layer 3 (sprite_registry.get_sprite): returns _FALLBACK instead of raising
#
# IDEMPOTENT — safe to call on every Streamlit rerun (checks existence first).
# Does zero network I/O.  Only writes files when something is missing.
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_sprite_assets() -> None:
    """
    Guarantee the minimum sprite asset infrastructure before registry init.

    Creates (if missing):
      • assets/sprites/{ani,ani-back,gen5,gen5-back,fallback}/  (directories)
      • assets/sprites/fallback/unknown.png                      (1×1 PNG)
      • assets/sprites/sprite_index.json                         (valid JSON)

    The generated index is built by scanning what files actually exist on disk.
    If no sprites have been downloaded the index has an empty ``sprites`` dict —
    init_sprite_registry() loads it successfully and every get_sprite() call
    returns the _FALLBACK singleton.

    When sprites ARE downloaded (by running scripts/download_sprites.py), the
    downloader regenerates the index with real paths and re-initialisation picks
    them up on the next server start.
    """
    import json as _json
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    sprite_dir = _Path(FALLBACK_PATH).parent.parent   # …/assets/sprites

    # ── 1. Create all required subdirectories ─────────────────────────────────
    for _sub in (
        "ani", "ani-back", "gen5", "gen5-back", "fallback",
        # Shiny variants — parallel subdirs, same filename convention
        "ani-shiny", "ani-back-shiny", "gen5-shiny", "gen5-back-shiny",
    ):
        (sprite_dir / _sub).mkdir(parents=True, exist_ok=True)

    # ── 2. Ensure fallback/unknown.png exists ─────────────────────────────────
    # Minimal valid PNG: 1×1 transparent RGBA pixel.  68 bytes.  No dependencies.
    _TINY_PNG = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR length + type
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # width=1, height=1
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,  # 8-bit RGBA, CRC
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT length + type
        0x54, 0x78, 0x9C, 0x62, 0x00, 0x00, 0x00, 0x02,  # compressed pixel
        0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00,  # IDAT CRC
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,  # IEND
        0x60, 0x82,
    ])
    _fb = _Path(FALLBACK_PATH)
    if not _fb.is_file() or _fb.stat().st_size == 0:
        try:
            _fb.write_bytes(_TINY_PNG)
        except OSError as _e:
            _log.warning("[bootstrap] Could not write fallback PNG: %s", _e)

    # ── 3. Generate sprite_index.json if missing ───────────────────────────────
    _index_path = _Path(DEFAULT_INDEX_PATH)
    if _index_path.is_file() and _index_path.stat().st_size > 10:
        return  # already exists and non-trivial — skip generation

    # Scan disk: include normal AND shiny sprites that were already downloaded.
    _TYPE_SCAN = [
        ("ani",            ".gif"),
        ("ani-back",       ".gif"),
        ("gen5",           ".png"),
        ("gen5-back",      ".png"),
        ("ani-shiny",      ".gif"),
        ("ani-back-shiny", ".gif"),
        ("gen5-shiny",     ".png"),
        ("gen5-back-shiny",".png"),
    ]
    _sprites: dict[str, list[str]] = {}
    for _type_key, _ext in _TYPE_SCAN:
        _type_dir = sprite_dir / _type_key
        if not _type_dir.is_dir():
            continue
        for _f in _type_dir.iterdir():
            if _f.is_file() and _f.suffix.lower() == _ext and _f.stat().st_size > 0:
                _sprites.setdefault(_f.stem, []).append(_type_key)

    # Write a valid index (sprites dict may be empty — that is correct and safe).
    _payload = {
        "version":      1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count":        len(_sprites),
        "sprites":      _sprites,
    }
    try:
        _index_path.write_text(
            _json.dumps(_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log.info(
            "[bootstrap] Generated sprite_index.json — %d slugs indexed "
            "(0 = no sprites downloaded yet; fallback active for all lookups).",
            len(_sprites),
        )
    except OSError as _e:
        _log.warning("[bootstrap] Could not write sprite_index.json: %s", _e)


# Run bootstrap synchronously at module load — BEFORE _init_sprites() is called.
# This is not inside @st.cache_resource because it does lightweight file I/O
# that must complete before the cached init can be attempted.
_bootstrap_sprite_assets()


# ── Sprite registry initialization ───────────────────────────────────────────
# @st.cache_resource ensures init_sprite_registry() runs exactly ONCE per
# Streamlit server process.  _bootstrap_sprite_assets() (above) guarantees
# sprite_index.json and fallback/unknown.png exist before this is called.

@st.cache_resource(show_spinner=False)
def _init_sprites(index_mtime: float = 0.0) -> RegistryMeta | None:
    """
    Initialize the sprite registry.  Re-runs automatically when sprite_index.json
    is updated (e.g. after running download_sprites.py --megas-pokeapi).

    The index_mtime argument is the file's modification timestamp — passing it
    as a parameter means @st.cache_resource uses it as part of the cache key.
    When the file is updated, mtime changes → cache miss → registry reloads.
    This avoids needing to restart Streamlit after downloading new sprites.
    """
    try:
        return init_sprite_registry(DEFAULT_INDEX_PATH)
    except FileNotFoundError:
        _log.error(
            "[_init_sprites] sprite_index.json still missing after bootstrap "
            "— get_sprite() will return FALLBACK for all lookups."
        )
        return None
    except Exception as _exc:
        _log.error("[_init_sprites] Unexpected error: %s: %s", type(_exc).__name__, _exc)
        return None


# Pass the current mtime of the index file so the registry auto-reloads
# whenever download_sprites.py updates it (no Streamlit restart needed).
_index_mtime: float = (
    os.path.getmtime(DEFAULT_INDEX_PATH)
    if os.path.exists(DEFAULT_INDEX_PATH) else 0.0
)
_SPRITE_META: RegistryMeta | None = _init_sprites(_index_mtime)


def format_catalog_label(value):
    return " ".join(part.capitalize() for part in str(value).replace("-", " ").strip().split())


@st.cache_data
def get_pokemon_catalog():
    """
    Return a sorted list of BASE Pokémon species display names.

    Two-layer defence against form variants entering the catalog:

    Layer 1 (primary) — endpoint choice
        Uses ``/pokemon-species`` instead of ``/pokemon``.
        The species endpoint returns ONLY canonical base species (≈1,021 entries
        as of Gen IX).  The plain ``/pokemon`` endpoint also returns every form
        variant ("charizard-mega-x", "raichu-alola", "pikachu-gmax" …), which
        would break sprite loading, team building, and battle initialisation.

    Layer 2 (secondary) — is_base_pokemon() filter
        Belt-and-suspenders guard in case the API response ever includes an
        unexpected entry, or the fallback list gains a stale form name.

    Display names are title-cased with hyphens replaced by spaces so the UI
    shows "Mr Mime" instead of "mr-mime".
    """
    # Offline catalog — derived from _LOCAL_POKEMON_DB keys.
    # format_catalog_label() applied so it matches the online catalog format.
    _FALLBACK = sorted(
        format_catalog_label(slug)
        for slug in _LOCAL_POKEMON_DB
        if is_base_pokemon(slug)
    )
    try:
        # /pokemon-species → base species only; no form entries
        response = requests.get(
            "https://pokeapi.co/api/v2/pokemon-species?limit=1500", timeout=10
        )
        response.raise_for_status()
        payload = response.json()
        raw_names = [entry["name"] for entry in payload.get("results", [])]
        # Secondary filter: guarantee no form slips through
        base_names = [n for n in raw_names if is_base_pokemon(n)]
        return [format_catalog_label(n) for n in base_names]
    except Exception:
        return sorted(
            n for n in _FALLBACK if is_base_pokemon(n)
        )


@st.cache_data
def get_item_catalog():
    fallback = ["Life Orb", "Leftovers", "Choice Band", "Choice Specs", "Focus Sash", "Assault Vest"]
    try:
        response = requests.get("https://pokeapi.co/api/v2/item?limit=2500", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [format_catalog_label(entry["name"]) for entry in payload.get("results", [])]
    except Exception:
        return sorted(set(fallback))


@st.cache_data
def get_item_data(item_name):
    if not item_name:
        return None
    try:
        url = f"https://pokeapi.co/api/v2/item/{item_name.lower().replace(' ', '-')}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        return {"name": payload["name"].capitalize(), "sprite": payload["sprites"]["default"]}
    except Exception:
        return None


# ── Sprite system ─────────────────────────────────────────────────────────────
# All sprite logic lives in src/sprite_registry.py.
# The dashboard is a pure renderer: it calls get_sprite() and passes the result
# to st.image().  It never loads sprites, never hits a CDN, never decides which
# file to use.
#
# get_sprite() is already O(1) — no @st.cache_data wrapper needed here.
# If the registry was not initialized (download script hasn't been run), the
# registry returns the local fallback placeholder for every lookup. No HTTP.


# ── Pokémon Showdown sprite CDN base URL ──────────────────────────────────────
# All sprites are served from play.pokemonshowdown.com.  No API key required.
# Animated GIF (ani/)  preferred; PNG (gen5/) used as fallback.
# These URLs are stable and work without any local download step.
SHOWDOWN_BASE = "https://play.pokemonshowdown.com/sprites"


def _showdown_slug(form_name: str) -> str:
    """
    Normalize a Pokémon/form name to the Pokémon Showdown CDN file slug.

    Delegates to normalize_showdown_name() from sprite_registry so the CDN
    URL uses the exact same slug as the local sprite filenames — a single
    source of truth for all Pokémon name → slug conversions.
    """
    return normalize_showdown_name(form_name)


def get_showdown_sprite(form_name: str, *, shiny: bool = False) -> dict[str, str]:
    """
    Build Pokémon Showdown CDN sprite URLs for *form_name*.

    Priority chain (highest quality first):
      front : animated GIF (ani/)  → static PNG (gen5/)
      back  : animated GIF (ani-back/) → static PNG (gen5-back/)

    For shiny Pokémon the same chain applies under the shiny subdirs.

    Parameters
    ----------
    form_name : str   e.g. "charizard", "charizard-mega-x", "pikachu-gmax"
    shiny     : bool  True to return shiny variants

    Returns
    -------
    dict with keys "front", "back", "ani" — all HTTPS CDN URLs.
    Never raises.  Never does local I/O.
    """
    slug = _showdown_slug(form_name)

    if shiny:
        ani_front = f"{SHOWDOWN_BASE}/ani-shiny/{slug}.gif"
        ani_back  = f"{SHOWDOWN_BASE}/ani-back-shiny/{slug}.gif"
        png_front = f"{SHOWDOWN_BASE}/gen5-shiny/{slug}.png"
        png_back  = f"{SHOWDOWN_BASE}/gen5-back-shiny/{slug}.png"
    else:
        ani_front = f"{SHOWDOWN_BASE}/ani/{slug}.gif"
        ani_back  = f"{SHOWDOWN_BASE}/ani-back/{slug}.gif"
        png_front = f"{SHOWDOWN_BASE}/gen5/{slug}.png"
        png_back  = f"{SHOWDOWN_BASE}/gen5-back/{slug}.png"

    return {
        "front": ani_front,   # animated preferred
        "back":  ani_back,
        "ani":   ani_front,
        # Also expose static PNG for contexts that can't render GIFs
        "front_png": png_front,
        "back_png":  png_back,
    }


# _ensure_fallback_sprite() was merged into _bootstrap_sprite_assets() above.
# Both the fallback PNG and the sprite index are created there, before
# _init_sprites() runs.  No separate call needed here.


def _is_url(path: str) -> bool:
    """Return True if *path* is an HTTP/HTTPS URL (not a local file path)."""
    return str(path).startswith(("http://", "https://"))


def _safe_sprite(data: dict, side: str = "front", shiny: bool | None = None) -> str:
    """
    Return a valid sprite path or URL for a Pokémon, respecting shiny state
    and the current active form (mega/gmax).

    Priority chain (highest → lowest):
      1. Local registry hit  — shiny + form  (only when sprites are downloaded)
      2. Local registry hit  — shiny + base  (only when sprites are downloaded)
      3. Local registry hit  — normal form   (only when sprites are downloaded)
      4. Local registry hit  — normal base   (only when sprites are downloaded)
      5. sprite_front/back Showdown CDN URL stored in the data dict
      6. Fresh Showdown CDN URL built from form/species/name
      7. FALLBACK_PATH       (1×1 transparent PNG — always exists locally)

    Parameters
    ----------
    data  : Pokémon state dict.  Expected keys:
              "form"         — active form slug (e.g. "charizard-mega-x")
              "species"      — base species slug (e.g. "charizard")
              "shiny"        — bool, from roll_shiny() at reset time
              "sprite_front" — Showdown URL (from get_pokemon_data)
              "sprite_back"  — Showdown URL (from get_pokemon_data)
    side  : "front" or "back".
    shiny : Override the data dict's shiny flag.  Pass None (default) to
            read shiny from data["shiny"].
    """
    is_shiny  = data.get("shiny", False) if shiny is None else shiny
    form_slug = (
        data.get("form")
        or data.get("species")
        or data.get("name", "")
    )

    # ── Priority 1-4: local registry (fast O(1), only useful when sprites
    #    have been downloaded with scripts/download_sprites.py) ──────────────
    if form_slug and is_initialized():
        paths = get_sprite(form_slug, shiny=is_shiny)
        path  = paths.front if side == "front" else paths.back
        if os.path.isfile(path):
            return path

    # ── Priority 5: Showdown CDN URL cached in the data dict ─────────────────
    key = "sprite_front" if side == "front" else "sprite_back"
    cached = data.get(key, "")
    if cached and _is_url(cached):
        return cached          # URL — Streamlit st.image() accepts these fine

    # ── Priority 6: Build a fresh Showdown CDN URL on-the-fly ─────────────────
    if form_slug:
        sprites = get_showdown_sprite(form_slug, shiny=is_shiny)
        return sprites["front"] if side == "front" else sprites["back"]

    # ── Priority 7: last resort local fallback ────────────────────────────────
    return FALLBACK_PATH


# ── Inline-HTML sprite helper ─────────────────────────────────────────────────
# st.html() and st.markdown(unsafe_allow_html=True) embed raw HTML that is
# rendered by the BROWSER.  The browser cannot fetch absolute server-side
# filesystem paths (e.g. /home/user/project/assets/sprites/front/pikachu.png).
# It interprets them as URL paths and gets a 404 from the Streamlit server.
#
# The only correct approach: read the file and embed it as a base64 data URI.
# This is cached per-path so each unique file is read from disk exactly once.
# ─────────────────────────────────────────────────────────────────────────────

# Transparent 1×1 pixel — returned when the file is missing or unreadable.
_DATA_URI_PLACEHOLDER = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIA"
    "BQABNjN9GQAAAAAAAAAAAAAAAAAAAAAAAAAAAACcAB7IAAAA"
)


@st.cache_data(show_spinner=False, max_entries=1024)
def _path_to_data_uri(path: str) -> str:
    """
    Convert *path* to an image source usable inside HTML rendered by the browser.

    Two cases:
    • HTTPS/HTTP URL  → returned as-is.  The browser fetches it directly from
      the CDN (Pokémon Showdown).  No server-side I/O.  This is the normal path
      when sprites come from get_showdown_sprite().
    • Local file path → read, base64-encode, return a data URI.  Used when
      sprites have been downloaded locally and the registry returns file paths.

    Safe to call from inside st.html() / st.markdown() blocks because the
    result is always a URL the browser can dereference — never a raw filesystem
    path that the browser would fail to load.

    Returns the transparent-pixel placeholder if the file is missing or
    unreadable (never raises).
    """
    if not path:
        return _DATA_URI_PLACEHOLDER
    # URLs go straight to the browser — no local I/O needed.
    if _is_url(path):
        return path
    # Local file → embed as base64 data URI.
    if not os.path.isfile(path):
        return _DATA_URI_PLACEHOLDER
    ext  = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "gif": "image/gif",
            "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    try:
        with open(path, "rb") as _f:
            encoded = base64.b64encode(_f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return _DATA_URI_PLACEHOLDER


@st.cache_data
def get_move_data(move_name):
    if not move_name:
        return None
    try:
        url = f"https://pokeapi.co/api/v2/move/{move_name.lower().replace(' ', '-').strip()}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        return {
            "name": format_name(payload["name"]),
            "api_name": payload["name"],
            "type": payload["type"]["name"],
            "power": payload["power"] or 0,
            "accuracy": payload["accuracy"],
            "pp": payload["pp"],
            "damage_class": payload["damage_class"]["name"],
            "target": payload["target"]["name"],
            "stat_changes": [
                {"name": entry["stat"]["name"], "change": entry["change"]}
                for entry in payload.get("stat_changes", [])
            ],
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  OFFLINE POKÉMON DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# Used when PokeAPI is unreachable (proxy block, no internet, rate-limit).
# Format per entry: (types_list, hp, atk, def, sp_atk, sp_def, spd)
# Stats from Bulbapedia — base stats, not EVs/IVs.
#
# Covers: all five competitive presets, both hardcoded default teams,
# the Pokémon in the random fallback catalog, and common fan favourites.
# Anything not listed falls back to the GENERIC template (70 across the board).
# ─────────────────────────────────────────────────────────────────────────────
_LOCAL_POKEMON_DB: dict[str, tuple] = {
    # Starters / fan favourites
    "bulbasaur":   (["grass", "poison"],          45,  49,  49,  65,  65,  45),
    "ivysaur":     (["grass", "poison"],          60,  62,  63,  80,  80,  60),
    "venusaur":    (["grass", "poison"],          80,  82,  83, 100, 100,  80),
    "charmander":  (["fire"],                     39,  52,  43,  60,  50,  65),
    "charmeleon":  (["fire"],                     58,  64,  58,  80,  65,  80),
    "charizard":   (["fire", "flying"],           78,  84,  78, 109,  85, 100),
    "squirtle":    (["water"],                    44,  48,  65,  50,  64,  43),
    "wartortle":   (["water"],                    59,  63,  80,  65,  80,  58),
    "blastoise":   (["water"],                    79,  83, 100,  85, 105,  78),
    "mewtwo":      (["psychic"],                 106, 110,  90, 154,  90, 130),
    "mew":         (["psychic"],                 100, 100, 100, 100, 100, 100),
    "pikachu":     (["electric"],                 35,  55,  40,  50,  50,  90),
    "raichu":      (["electric"],                 60,  90,  55,  90,  80, 110),
    "eevee":       (["normal"],                   55,  55,  50,  45,  65,  55),
    "vaporeon":    (["water"],                   130,  65,  60, 110,  95,  65),
    "jolteon":     (["electric"],                 65,  65,  60, 110,  95, 130),
    "flareon":     (["fire"],                     65, 130,  60,  95,  110,  65),
    "espeon":      (["psychic"],                  65,  65,  60, 130,  95, 110),
    "umbreon":     (["dark"],                     95,  65, 110,  60, 130,  65),
    "leafeon":     (["grass"],                    65, 110, 130,  60,  65,  95),
    "glaceon":     (["ice"],                      65,  60, 110, 130,  95,  65),
    "sylveon":     (["fairy"],                    95,  65,  65, 110, 130,  60),
    # Pseudo-legendaries
    "dragonite":   (["dragon", "flying"],         91, 134,  95, 100, 100,  80),
    "tyranitar":   (["rock", "dark"],            100, 134, 110,  95, 100,  61),
    "salamence":   (["dragon", "flying"],         95, 135,  80, 110,  80, 100),
    "metagross":   (["steel", "psychic"],         80, 135, 130,  95,  90,  70),
    "garchomp":    (["dragon", "ground"],        108, 130,  95,  80,  85, 102),
    "hydreigon":   (["dark", "dragon"],           92, 105,  90, 125,  90,  98),
    "goodra":      (["dragon"],                   90,  100,  70, 110, 150,  80),
    "kommo-o":     (["dragon", "fighting"],       75, 110, 125, 100,  105, 85),
    # Box legendaries / Ubers
    "articuno":    (["ice", "flying"],            90,  85, 100,  95, 125,  85),
    "zapdos":      (["electric", "flying"],       90, 100,  85, 125,  90, 100),
    "moltres":     (["fire", "flying"],           90, 100,  90, 125,  85,  90),
    "raikou":      (["electric"],                 90,  85,  75, 115, 100, 115),
    "entei":       (["fire"],                    115, 115,  85,  90,  75, 100),
    "suicune":     (["water"],                   100,  75, 115,  90, 115,  85),
    "lugia":       (["psychic", "flying"],       106,  90, 130,  90, 154,  110),
    "ho-oh":       (["fire", "flying"],          106, 130,  90, 110, 154,  90),
    "celebi":      (["psychic", "grass"],        100, 100, 100, 100, 100, 100),
    "rayquaza":    (["dragon", "flying"],        105, 150,  90, 150,  90,  95),
    "kyogre":      (["water"],                   100, 100,  90, 150, 140,  90),
    "groudon":     (["ground"],                  100, 150, 140, 100,  90,  90),
    "latias":      (["dragon", "psychic"],        80,  80,  90, 110, 130, 110),
    "latios":      (["dragon", "psychic"],        80,  90,  80, 130, 110, 110),
    "jirachi":     (["steel", "psychic"],        100, 100, 100, 100, 100, 100),
    "deoxys":      (["psychic"],                  50, 150,  50, 150,  50, 150),
    "dialga":      (["steel", "dragon"],         100, 120, 120, 150, 100,  90),
    "palkia":      (["water", "dragon"],          90, 120, 100, 150, 120, 100),
    "giratina":    (["ghost", "dragon"],         150, 100, 120, 100, 120,  90),
    "arceus":      (["normal"],                  120, 120, 120, 120, 120, 120),
    "reshiram":    (["dragon", "fire"],          100, 120, 100, 150, 120,  90),
    "zekrom":      (["dragon", "electric"],      100, 150, 120, 120, 100,  90),
    "kyurem":      (["dragon", "ice"],           125, 130,  90, 130,  90,  95),
    "xerneas":     (["fairy"],                   126, 131,  95, 131,  98,  99),
    "yveltal":     (["dark", "flying"],          126, 131,  95, 131,  98,  99),
    "zygarde":     (["dragon", "ground"],        108, 100, 121,  81,  95,  95),
    # Competitive staples
    "alakazam":    (["psychic"],                  55,  50,  45, 135,  95, 120),
    "gengar":      (["ghost", "poison"],          60,  65,  60, 130,  75, 110),
    "machamp":     (["fighting"],                 90, 130,  80,  65,  85,  55),
    "golem":       (["rock", "ground"],           80, 120, 130,  55,  65,  45),
    "slowbro":     (["water", "psychic"],         95,  75, 110, 100,  80,  30),
    "cloyster":    (["water", "ice"],             50,  95, 180,  85,  45,  70),
    "starmie":     (["water", "psychic"],         60,  75,  85, 100,  85, 115),
    "scyther":     (["bug", "flying"],            70, 110,  80,  55,  80, 105),
    "scizor":      (["bug", "steel"],             70, 130, 100,  55,  80,  65),
    "gyarados":    (["water", "flying"],          95, 125,  79,  60, 100,  81),
    "lapras":      (["water", "ice"],            130,  85,  80,  85,  95,  60),
    "snorlax":     (["normal"],                  160, 110,  65,  65, 110,  30),
    "tauros":      (["normal"],                   75, 100,  95,  40,  70, 110),
    "porygon":     (["normal"],                   65,  60,  70,  85,  75,  40),
    "porygon2":    (["normal"],                   85,  80,  90, 105,  95,  60),
    "porygon-z":   (["normal"],                   85,  80,  70, 135,  75,  90),
    "magmar":      (["fire"],                     65,  95,  57, 100,  85,  93),
    "electabuzz":  (["electric"],                 65,  83,  57,  95,  85, 105),
    "jynx":        (["ice", "psychic"],           65,  50,  35, 115,  95,  95),
    "aerodactyl":  (["rock", "flying"],           80, 105,  65,  60,  75, 130),
    "exeggutor":   (["grass", "psychic"],         95,  95,  85, 125,  65,  55),
    "clefable":    (["fairy"],                    95,  70,  73,  95,  90,  60),
    "wigglytuff":  (["normal", "fairy"],         140,  70,  45,  85,  50,  45),
    "togekiss":    (["fairy", "flying"],          85,  50,  95, 120, 115,  80),
    "togepi":      (["fairy"],                    35,  20,  65,  40,  65,  20),
    # Water types
    "swampert":    (["water", "ground"],         100, 110,  90,  85,  90,  60),
    "feraligatr":  (["water"],                    85, 105, 100,  79,  83,  78),
    "politoed":    (["water"],                    90,  75,  75,  90, 100,  70),
    "vaporeon":    (["water"],                   130,  65,  60, 110,  95,  65),
    "kingdra":     (["water", "dragon"],          75,  95,  95,  95,  95,  85),
    "kabutops":    (["rock", "water"],            60, 115, 105,  65,  70,  80),
    "clawitzer":   (["water"],                    71,  73,  88, 120,  89,  59),
    "milotic":     (["water"],                    95,  60,  79, 100, 125,  81),
    # Fire types
    "arcanine":    (["fire"],                     90, 110,  80, 100,  80,  95),
    "ninetales":   (["fire"],                     73,  76,  75,  81, 100, 100),
    "heatran":     (["fire", "steel"],            91,  90, 106, 130, 106,  77),
    "volcarona":   (["bug", "fire"],              85,  60,  65, 135, 105, 100),
    "chandelure":  (["ghost", "fire"],            60,  55,  90, 145,  90,  80),
    "incineroar":  (["fire", "dark"],             95, 115,  90,  80,  90,  60),
    "talonflame":  (["fire", "flying"],           78,  81,  71,  74,  69, 126),
    # Electric types
    "jolteon":     (["electric"],                 65,  65,  60, 110,  95, 130),
    "magnezone":   (["electric", "steel"],        70, 70, 115, 130,  90,  60),
    "rotom":       (["electric", "ghost"],        50,  65, 107,  105,  107,  86),
    # Grass types
    "venusaur":    (["grass", "poison"],          80,  82,  83, 100, 100,  80),
    "sceptile":    (["grass"],                    70,  85,  65, 105,  85, 120),
    "roserade":    (["grass", "poison"],          60,  70,  65, 125, 105,  90),
    "victreebel":  (["grass", "poison"],          80, 105,  65, 100,  60,  70),
    "ludicolo":    (["water", "grass"],           80,  70,  70,  90, 100,  70),
    # Ice types
    "mamoswine":   (["ice", "ground"],           110, 130,  80,  70,  60,  80),
    "weavile":     (["dark", "ice"],              70, 120,  65,  45,  85, 125),
    "glaceon":     (["ice"],                      65,  60, 110, 130,  95,  65),
    "froslass":    (["ice", "ghost"],             70,  80,  70,  80,  70, 110),
    "cloyster":    (["water", "ice"],             50,  95, 180,  85,  45,  70),
    # Fighting types
    "lucario":     (["fighting", "steel"],        70, 110,  70, 115,  70,  90),
    "machamp":     (["fighting"],                 90, 130,  80,  65,  85,  55),
    "toxicroak":   (["poison", "fighting"],       83, 106,  65,  86,  65,  85),
    "pangoro":     (["fighting", "dark"],         95, 124,  78,  69,  71,  58),
    "conkeldurr":  (["fighting"],                105, 140,  95,  55,  65,  45),
    "infernape":   (["fire", "fighting"],         76, 104,  71, 104,  71, 108),
    # Psychic types
    "alakazam":    (["psychic"],                  55,  50,  45, 135,  95, 120),
    "espeon":      (["psychic"],                  65,  65,  60, 130,  95, 110),
    "gardevoir":   (["psychic", "fairy"],         68,  65,  65, 125,  115,  80),
    "reuniclus":   (["psychic"],                  110,  65,  75, 125,  85,  30),
    "slowbro":     (["water", "psychic"],         95,  75, 110, 100,  80,  30),
    # Ghost types
    "gengar":      (["ghost", "poison"],          60,  65,  60, 130,  75, 110),
    "haunter":     (["ghost", "poison"],          45,  50,  45, 115,  55,  95),
    "mismagius":   (["ghost"],                    60,  60,  60, 105, 105, 105),
    "chandelure":  (["ghost", "fire"],            60,  55,  90, 145,  90,  80),
    # Dark types
    "tyranitar":   (["rock", "dark"],            100, 134, 110,  95, 100,  61),
    "absol":       (["dark"],                     65, 130,  60,  75,  60,  75),
    "bisharp":     (["dark", "steel"],            65, 125, 100,  60,  70,  70),
    "weavile":     (["dark", "ice"],              70, 120,  65,  45,  85, 125),
    "krookodile":  (["ground", "dark"],           95, 117,  80,  65,  70,  92),
    "hydreigon":   (["dark", "dragon"],           92, 105,  90, 125,  90,  98),
    "incineroar":  (["fire", "dark"],             95, 115,  90,  80,  90,  60),
    # Steel types
    "scizor":      (["bug", "steel"],             70, 130, 100,  55,  80,  65),
    "metagross":   (["steel", "psychic"],         80, 135, 130,  95,  90,  70),
    "heatran":     (["fire", "steel"],            91,  90, 106, 130, 106,  77),
    "lucario":     (["fighting", "steel"],        70, 110,  70, 115,  70,  90),
    "aegislash":   (["steel", "ghost"],           60, 50,  150,  50, 150,  60),
    "excadrill":   (["ground", "steel"],         110, 135,  60,  50,  65,  88),
    # Ground types
    "garchomp":    (["dragon", "ground"],        108, 130,  95,  80,  85, 102),
    "groudon":     (["ground"],                  100, 150, 140, 100,  90,  90),
    "swampert":    (["water", "ground"],         100, 110,  90,  85,  90,  60),
    "excadrill":   (["ground", "steel"],         110, 135,  60,  50,  65,  88),
    "krookodile":  (["ground", "dark"],           95, 117,  80,  65,  70,  92),
    "mamoswine":   (["ice", "ground"],           110, 130,  80,  70,  60,  80),
    # Rock / misc
    "tyranitar":   (["rock", "dark"],            100, 134, 110,  95, 100,  61),
    "aerodactyl":  (["rock", "flying"],           80, 105,  65,  60,  75, 130),
    "kabutops":    (["rock", "water"],            60, 115, 105,  65,  70,  80),
    # Normal / misc
    "snorlax":     (["normal"],                  160, 110,  65,  65, 110,  30),
    "blissey":     (["normal"],                  255,  10,  10,  75, 135,  55),
    "pidgeot":     (["normal", "flying"],         83,  80,  75,  70,  70, 101),
    "raticate":    (["normal"],                   55,  81,  60,  50,  70,  97),
    "togekiss":    (["fairy", "flying"],          85,  50,  95, 120, 115,  80),
}


def _build_fallback_pokemon(slug: str, display_name: str, item_name: str) -> dict:
    """
    Build a valid Pokémon data dict from local data when PokeAPI is unavailable.

    Priority:
      1. _LOCAL_POKEMON_DB entry (accurate Bulbapedia stats + real types)
      2. Generic template (70 across the board, Normal type) — always available

    The returned dict is structurally identical to what get_pokemon_data() would
    return from a successful API call: all downstream code (generate_moveset,
    _apply_form_transforms, BattleEngine) works without changes.

    The "_is_fallback": True flag is a diagnostic marker — the dashboard
    renders a subtle offline badge on the Pokémon card.
    """
    entry = _LOCAL_POKEMON_DB.get(slug)
    if entry:
        types, hp, atk, def_, sp_atk, sp_def, spd = entry
    else:
        # Generic template for any Pokémon not in the local DB.
        # Balanced stats so it's playable; Normal type is the safest default.
        types   = ["normal"]
        hp      = 75
        atk     = 80
        def_    = 70
        sp_atk  = 80
        sp_def  = 70
        spd     = 75

    base_stats = {
        "hp":     hp,
        "atk":    atk,
        "def":    def_,
        "sp_atk": sp_atk,
        "sp_def": sp_def,
        "spd":    spd,
    }
    # Use Showdown CDN URLs so the fallback card shows a real sprite even
    # when the local sprite download hasn't been run.
    _showdown    = get_showdown_sprite(slug)
    sprite_front = _showdown["front"]
    sprite_back  = _showdown["back"]

    # Capitalise the display name for UI consistency
    nice_name = " ".join(p.capitalize() for p in display_name.replace("-", " ").split())

    return {
        "name":            nice_name,
        # ── Form / shiny state fields ─────────────────────────────────────────
        "species":         slug,
        "form":            slug,
        "shiny":           False,        # rolled by env.reset() / BattleEngine
        "mega_evolved":    False,
        # ── Sprite paths ──────────────────────────────────────────────────────
        "sprite_front":    sprite_front,
        "sprite_back":     sprite_back,
        "types":           types,
        "base_stats":      base_stats,
        "stats":           dict(base_stats),
        "stat_stages":     {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0},
        "current_hp":      1.0,
        "status":          None,
        "item":            None,        # item API also offline → skip gracefully
        "debilitado":      False,
        "role_info":       get_role_info(base_stats),
        # Empty list → generate_moveset() falls back to Struggle (battle still works)
        "_all_move_names": [],
        # Diagnostic flag — renders offline badge on the card
        "_is_fallback":    True,
    }


@st.cache_data
def get_pokemon_data(name_or_id, item_name="Life Orb"):
    """
    Fetch BASE Pokémon data from PokeAPI, with a full offline fallback.

    ──────────────────────────────────────────────────────────────────────────
    NETWORK STRATEGY
    ──────────────────────────────────────────────────────────────────────────
    1. Try PokeAPI (online path):
       • Slug-normalise the name ("Mr Mime" → "mr-mime")
       • GET /pokemon/<slug> with 5 s timeout
       • Parse stats, types, sprites, move list
    2. On ANY network failure (ProxyError, timeout, 404, SSL, …):
       • Log the exact exception type and message to the terminal
       • Call _build_fallback_pokemon() → returns accurate local stats for
         ~130 known Pokémon or balanced generic stats for unknowns
       • NEVER returns None

    This means the app is fully functional offline: teams always have 6
    members, INICIAR COMBATE always passes validation, and battles can start.

    Cache key: (name_or_id, item_name)

    The private field "_all_move_names" carries the raw list of API move-name
    strings for generate_moveset().  In offline mode it is [] → Struggle fills
    all 4 move slots, which is playable.
    """
    # Normalise display name → PokeAPI slug.
    # "Mr Mime" → "mr-mime", "Tapu Koko" → "tapu-koko", "Charizard" → "charizard"
    _slug = str(name_or_id).lower().strip().replace(" ", "-")
    _url  = f"https://pokeapi.co/api/v2/pokemon/{_slug}"

    try:
        print(f"[get_pokemon_data] GET {_url}")          # visible in server terminal
        response = requests.get(_url, timeout=5)
        print(f"[get_pokemon_data] HTTP {response.status_code} for '{_slug}'")
        response.raise_for_status()
        payload = response.json()

        api_stats    = {e["stat"]["name"]: e["base_stat"] for e in payload["stats"]}
        _api_name    = payload["name"]
        # Use Showdown CDN URLs — works without any local sprite download.
        # get_sprite() is also called so the local registry is consulted when
        # sprites have been downloaded; CDN URLs serve as the universal fallback.
        _showdown    = get_showdown_sprite(_api_name)

        base_stats = {
            "hp":     api_stats.get("hp", 0),
            "atk":    api_stats.get("attack", 0),
            "def":    api_stats.get("defense", 0),
            "sp_atk": api_stats.get("special-attack", 0),
            "sp_def": api_stats.get("special-defense", 0),
            "spd":    api_stats.get("speed", 0),
        }
        pokemon_types  = [e["type"]["name"] for e in payload["types"]]
        all_move_names = [e["move"]["name"] for e in payload.get("moves", [])]

        return {
            "name":            payload["name"].capitalize(),
            # ── Form / shiny state fields ─────────────────────────────────────
            # species is set here; form starts as species and may be updated by
            # BattleEngine._apply_item_transforms() at the start of turn 1.
            "species":         _api_name,
            "form":            _api_name,
            "shiny":           False,        # rolled by env.reset() / BattleEngine
            "mega_evolved":    False,
            # ── Sprite paths (Showdown CDN URLs — no local download needed) ───
            "sprite_front":    _showdown["front"],
            "sprite_back":     _showdown["back"],
            "types":           pokemon_types,
            "base_stats":      base_stats,
            "stats":           dict(base_stats),
            "stat_stages":     {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0},
            "current_hp":      1.0,
            "status":          None,
            "item":            get_item_data(item_name),
            "debilitado":      False,
            "role_info":       get_role_info(base_stats),
            "_all_move_names": all_move_names,
            "_is_fallback":    False,
        }

    except Exception as exc:
        # ── Log the EXACT failure reason ──────────────────────────────────────
        # ProxyError, ConnectionError, Timeout, HTTPError (404), SSLError, …
        # All are caught here; the terminal shows the full class name and message.
        print(
            f"[get_pokemon_data] OFFLINE fallback for '{_slug}': "
            f"{type(exc).__name__}: {exc}"
        )
        _log.warning(
            "get_pokemon_data(%r) failed — using local fallback. "
            "Reason: %s: %s",
            _slug, type(exc).__name__, exc,
        )
        # ── Return local data — NEVER None ────────────────────────────────────
        return _build_fallback_pokemon(_slug, str(name_or_id), item_name)


def generate_moveset(base_data: dict, moveset_mode: str) -> tuple[list, list]:
    """
    Build a moveset for a Pokémon from its base data dict.

    This function is deliberately NOT cached at the composite level — the
    individual get_move_data() calls it makes are already cached, so repeated
    invocations with the same arguments are cheap.

    Returns (moves, move_pool) where:
        moves     — list of exactly 4 move dicts for the active battle slot
        move_pool — list of up to 20 scored candidate moves for Custom editing

    STRICT ISOLATION: This function never calls get_pokemon_data().
    Changing moveset_mode triggers no PokeAPI Pokémon fetches whatsoever.
    """
    all_move_names = base_data.get("_all_move_names", [])
    pokemon_types  = base_data["types"]
    base_stats     = base_data["base_stats"]
    pokemon_name   = base_data.get("name", "").lower()

    _struggle = {
        "name": "Struggle", "api_name": "struggle",
        "type": "normal", "power": 50, "accuracy": None,
        "pp": 1, "damage_class": "physical",
        "target": "selected-pokemon", "stat_changes": [],
    }

    # ── "random" mode: first 4 fetchable moves in PokeAPI order ──────────────
    if moveset_mode == "random":
        raw_moves: list[dict] = []
        for mn in all_move_names:
            md = get_move_data(mn)
            if md:
                raw_moves.append(md)
            if len(raw_moves) == 4:
                break
        while len(raw_moves) < 4:
            raw_moves.append(dict(_struggle))
        return raw_moves, raw_moves

    # ── competitive / balanced / custom: use the intelligent pipeline ─────────
    # Step 1: heuristic prefilter on name alone (no API calls)
    candidate_names = prefilter_move_names(all_move_names, pokemon_types, limit=22)

    # Step 2: fetch move details (each call is cached by get_move_data)
    candidates: list[dict] = []
    for mn in candidate_names:
        md = get_move_data(mn)
        if md:
            candidates.append(md)

    if not candidates:
        return [dict(_struggle)] * 4, []

    # Step 3: select final moveset
    # "custom" uses the same algorithm as "competitive" but also exposes the
    # full pool so the UI can render edit dropdowns.
    build_mode = "competitive" if moveset_mode == "custom" else moveset_mode
    moves = build_moveset(pokemon_name, pokemon_types, base_stats, candidates, mode=build_mode)

    # Step 4: build the pool for custom editing
    move_pool = get_filtered_move_pool(candidates, pokemon_types, base_stats, limit=20)

    return moves, move_pool


def reset_pokemon_state(pokemon):
    """Reset a Pokémon dict to full battle-start state (kept for offline use)."""
    pokemon["current_hp"] = 1.0
    pokemon["status"] = None
    pokemon["debilitado"] = False
    pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    pokemon["stats"] = apply_stat_stages(pokemon["base_stats"], pokemon["stat_stages"])


def _es() -> dict:
    """
    Return the engine's authoritative battle state dict.

    Architecture contract
    ---------------------
    BattleEngine is the SOLE SOURCE OF TRUTH for team data during combat.
    session_state NEVER holds a second copy of team_ia / team_rival —
    all rendering code reads directly from this call.

    This replaces the old _sync_state_from_engine() mirror pattern.
    No session_state writes are performed; no sync is needed.
    Only call this after INICIAR COMBATE (game_started == True).
    """
    return st.session_state.env.get_state()


# Known status / utility move effect descriptions used in tooltips.
_KNOWN_MOVE_EFFECTS: dict[str, str] = {
    # Stat boosts — self
    "swords-dance":     "Raises user's Attack by +2 stages.",
    "nasty-plot":       "Raises user's Sp. Atk by +2 stages.",
    "calm-mind":        "Raises user's Sp. Atk and Sp. Def by +1 stage each.",
    "bulk-up":          "Raises user's Attack and Defense by +1 stage each.",
    "dragon-dance":     "Raises user's Attack and Speed by +1 stage each.",
    "quiver-dance":     "Raises user's Sp. Atk, Sp. Def, and Speed by +1 stage each.",
    "shell-smash":      "Drops Defense and Sp. Def by -1; raises Attack, Sp. Atk, Speed by +2.",
    "growth":           "Raises Sp. Atk by +1 (or +2 in harsh sunlight).",
    "iron-defense":     "Raises user's Defense by +2 stages.",
    "amnesia":          "Raises user's Sp. Def by +2 stages.",
    "hone-claws":       "Raises user's Attack and Accuracy by +1 stage each.",
    "work-up":          "Raises user's Attack and Sp. Atk by +1 stage each.",
    "coil":             "Raises user's Attack, Defense, and Accuracy by +1 stage each.",
    # Stat drops — opponent
    "growl":            "Lowers the opponent's Attack by -1 stage.",
    "tail-whip":        "Lowers the opponent's Defense by -1 stage.",
    "leer":             "Lowers the opponent's Defense by -1 stage.",
    "screech":          "Lowers the opponent's Defense by -2 stages.",
    "fake-tears":       "Lowers the opponent's Sp. Def by -2 stages.",
    "metal-sound":      "Lowers the opponent's Sp. Def by -2 stages.",
    "charm":            "Lowers the opponent's Attack by -2 stages.",
    "sweet-scent":      "Lowers the opponent's Evasiveness by -1 stage.",
    "string-shot":      "Lowers the opponent's Speed by -1 stage.",
    "scary-face":       "Lowers the opponent's Speed by -2 stages.",
    # Status conditions
    "thunder-wave":     "Paralyzes the opponent. Reduces Speed by 50%.",
    "toxic":            "Badly poisons the opponent (damage worsens each turn).",
    "poison-powder":    "Poisons the opponent.",
    "sleep-powder":     "Puts the opponent to sleep.",
    "hypnosis":         "Puts the opponent to sleep (60% accuracy).",
    "spore":            "Puts the opponent to sleep (100% accuracy).",
    "stun-spore":       "Paralyzes the opponent.",
    "will-o-wisp":      "Burns the opponent, halving its Attack.",
    "glare":            "Paralyzes the opponent.",
    "sing":             "Puts the opponent to sleep (55% accuracy).",
    # Recovery / healing
    "recover":          "Restores up to 50% of user's max HP.",
    "softboiled":       "Restores up to 50% of user's max HP.",
    "roost":            "Restores up to 50% of user's max HP. Removes Flying type this turn.",
    "slack-off":        "Restores up to 50% of user's max HP.",
    "milk-drink":       "Restores up to 50% of user's max HP.",
    "synthesis":        "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "moonlight":        "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "morning-sun":      "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "rest":             "User sleeps for 2 turns and fully restores HP + cures status.",
    "wish":             "At end of next turn, the Pokémon on field restores 50% of max HP.",
    "aqua-ring":        "Surrounds user in water to restore a little HP each turn.",
    "ingrain":          "Roots user in ground; restores HP each turn but prevents switching.",
    # Hazards / field
    "stealth-rock":     "Lays rocks that deal type-based damage (6–50%) to switching-in Pokémon.",
    "spikes":           "Lays spikes that damage (non-Flying) Pokémon when they switch in.",
    "toxic-spikes":     "Poisons Pokémon that switch in (2 layers = badly poisoned).",
    "sticky-web":       "Lowers Speed of grounded Pokémon that switch in.",
    "defog":            "Clears hazards and screens for both sides.",
    "rapid-spin":       "Clears entry hazards from user's side. Power 50.",
    # Protection / utility
    "protect":          "User is protected from most moves this turn. Fails if used consecutively.",
    "detect":           "Same as Protect (uses a separate turn counter).",
    "endure":           "User always survives this turn with at least 1 HP.",
    "substitute":       "Creates a decoy using 25% of max HP.",
    "encore":           "Forces opponent to repeat its last move for 3 turns.",
    "taunt":            "Prevents opponent from using status moves for 3 turns.",
    "trick":            "Swaps held items with the opponent.",
    "switcheroo":       "Swaps held items with the opponent.",
    "knock-off":        "Removes opponent's held item; deals bonus damage if item is present.",
    "leech-seed":       "Seeds the opponent; drains 1/8 of its HP each turn to heal the user.",
    "rain-dance":       "Summons rain for 5 turns. Boosts Water moves, weakens Fire moves.",
    "sunny-day":        "Summons harsh sunlight for 5 turns. Boosts Fire moves, weakens Water.",
    "sandstorm":        "Summons a sandstorm for 5 turns. Damages non-Rock/Ground/Steel types.",
    "hail":             "Summons hail for 5 turns. Damages non-Ice types.",
    "trick-room":       "Slower Pokémon move first for 5 turns.",
    "reflect":          "Halves physical damage taken by user's side for 5 turns.",
    "light-screen":     "Halves special damage taken by user's side for 5 turns.",
    "aurora-veil":      "Requires hail; halves both physical and special damage for 5 turns.",
    "baton-pass":       "Switches out, passing any stat changes to the next Pokémon.",
    "u-turn":           "Deals damage then switches the user out. Power 70.",
    "volt-switch":      "Deals damage then switches the user out. Power 70.",
}


def _describe_status_move(move: dict) -> str:
    """Return an effect description for status / utility moves."""
    name = (move.get("name") or "").lower().strip()
    return _KNOWN_MOVE_EFFECTS.get(name, "")


def get_move_tooltip(move: dict, defender: dict) -> str:
    """Rich tooltip for a move button — includes effect descriptions for status moves."""
    type_name    = format_name(move.get("type"))
    damage_class = format_name(move.get("damage_class"))
    power        = move.get("power") or 0
    accuracy     = move.get("accuracy") or 100
    multiplier   = get_type_multiplier(move.get("type"), defender.get("types", []))
    eff_label    = describe_effectiveness(multiplier)

    parts = [
        f"Type: {type_name}",
        f"Class: {damage_class}",
        f"Power: {power or '—'}",
        f"Accuracy: {accuracy}%",
        f"Effectiveness: {eff_label}",
    ]

    # Append effect description for status/utility moves
    if not power:
        effect = _describe_status_move(move)
        if effect:
            parts.append(f"Effect: {effect}")

    return " | ".join(parts)


def _send_in_pokemon(side: str, new_idx: int) -> None:
    """
    Delegate a Pokémon switch-in to the engine, then sync session_state.

    The engine applies entry-hazard chip damage internally; we only need to
    surface any hazard log string to the battle history.
    """
    haz_log = st.session_state.env.send_in(side, new_idx)
    if haz_log:
        st.session_state.historial.insert(0, f"📌 {haz_log}")


def handle_post_turn_state() -> None:
    """
    Process post-turn faint events by delegating to the engine.

    The engine's handle_post_faint() decides whether to auto-advance or
    signal that the player must choose (challenge mode, rival side only).
    All team mutation happens inside the engine; this function only reads
    the result dict and updates UI-level session_state flags.
    """
    engine         = st.session_state.env
    challenge_mode = st.session_state.get("battle_mode", "1. Simulación") == "2. Desafío"

    # ── Rival Pokémon fainted ────────────────────────────────────────────────
    if engine.hp_rival <= 0:
        result = engine.handle_post_faint("rival", challenge_mode=challenge_mode)
        if result["haz_log"]:
            st.session_state.historial.insert(0, f"📌 {result['haz_log']}")
        if result["battle_over"]:
            st.session_state.battle_finished = True
            st.session_state.resultado = result["outcome"]
        elif result["must_choose"]:
            st.session_state.must_switch_rival = True

    # ── IA Pokémon fainted ───────────────────────────────────────────────────
    if engine.hp_ia <= 0:
        result = engine.handle_post_faint("ia", challenge_mode=False)
        if result["haz_log"]:
            st.session_state.historial.insert(0, f"📌 {result['haz_log']}")
        if result["battle_over"]:
            st.session_state.battle_finished = True
            st.session_state.resultado = result["outcome"]


# ─────────────────────────────────────────────────────────────────────────────
# FORM PIPELINE  (module-level — available to both battle and team-builder UIs)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ------------
# Layer 1 · get_pokemon_data()     → raw base data, no form changes, no moves
# Layer 2 · resolve_form()         → pure Python, deterministic, no API calls
#           (imported from src.pokemon_forms)
# Layer 3 · get_showdown_sprite()  → Showdown CDN with HEAD-verified fallback chain
# Layer 4 · _apply_form_transforms → assembles the final display dict
#
# NEVER mutate base_data.  Every function returns a fresh dict.
# ─────────────────────────────────────────────────────────────────────────────


def _form_display_name(capitalized_base: str, form_name: str) -> str:
    """
    Derive a human-readable display name from the PokeAPI form_name.

    Parameters
    ----------
    capitalized_base : str  e.g. "Charizard"
    form_name        : str  e.g. "charizard-mega-x"  (PokeAPI lowercase)

    Returns
    -------
    str  e.g. "Charizard-Mega-X"

    The algorithm strips the base prefix from form_name, then title-cases
    each hyphen-separated segment of the remaining suffix.
    """
    norm_base = capitalized_base.lower().replace(" ", "-")
    suffix    = form_name[len(norm_base):]           # e.g. "-mega-x"
    parts     = [p.capitalize() for p in suffix.split("-") if p]
    return capitalized_base + ("-" + "-".join(parts) if parts else "")


def _apply_form_transforms(
    data: dict,
    item_name: str,
    flags: dict | None = None,
) -> dict:
    """
    Return a new Pokémon display dict with all form-based overlays applied.

    The input *data* dict is **never** mutated — this function always builds
    and returns a fresh dict.

    Pipeline
    --------
    1. resolve_form()         — pure Python; explicit tables, no string hacks
    2. get_showdown_sprite()  — Showdown CDN, HEAD-verified, animated GIF preferred
    3. dict assembly          — scale stats, override types, set display name

    Supported form types
    --------------------
    mega     : Mega Evolution and Primal Reversion (item in MEGA_STONE_MAP)
    gmax     : Gigantamax (item == "Gigantamax Factor" OR flags["gmax"])
    dynamax  : Dynamax    (item == "Dynamax Band"    OR flags["dynamax"])
    base     : no transformation

    Parameters
    ----------
    data      : Pokémon dict as returned by get_pokemon_data() + generate_moveset()
    item_name : held item display name (e.g. "Charizardite X")
    flags     : optional battle-state overrides {"gmax": bool, "dynamax": bool}
    """
    # Gigantamax can also be expressed purely through the item name; promote it
    # to a flag so resolve_form() sees a consistent interface.
    item_key        = (item_name or "").lower().strip()
    effective_flags = dict(flags or {})
    if item_key in ("gigantamax factor", "gigantamax-factor"):
        effective_flags["gmax"] = True

    # ── Layer 2: resolve which form applies ──────────────────────────────────
    base_api_name = data["name"].lower().replace(" ", "-")
    form = resolve_form(base_api_name, item_name, effective_flags)

    # ── Layer 3: fetch Showdown sprites for this form ────────────────────────
    is_shiny = data.get("shiny", False)
    sprites = get_showdown_sprite(form["form_name"], shiny=is_shiny)

    # ── Scale stats ──────────────────────────────────────────────────────────
    scaled_stats = {k: int(v * form["stat_mult"]) for k, v in data["base_stats"].items()}
    if form["hp_mult"] != 1.0:
        scaled_stats["hp"] = int(scaled_stats["hp"] * form["hp_mult"])

    # ── Derive display name ───────────────────────────────────────────────────
    if form["form_type"] == "base":
        display_name = data["name"]
    else:
        display_name = _form_display_name(data["name"], form["form_name"])

    # ── Layer 4: assemble final dict (shallow copy — never mutates input) ─────
    return {
        **data,
        "name":         display_name,
        # "form" is read by _safe_sprite() to do the local-registry lookup.
        # Must be set to the ACTIVE form slug (e.g. "charizard-mega-x") so that
        # normalize_showdown_name() converts it to the merged slug "charizardmegax"
        # and finds the downloaded sprite in the local index.
        "form":         form["form_name"],
        "types":        form["types"] if form["types"] is not None else data["types"],
        "base_stats":   scaled_stats,
        "stats":        scaled_stats,
        "sprite_front": sprites["front"],
        "sprite_back":  sprites["back"],
        "_form":        form["form_type"],
        "_form_name":   form["form_name"],
    }


if "game_started" not in st.session_state:
    st.session_state.update(
        {
            "game_started": False,
            "battle_finished": False,
            "resultado": "",
            # active_ia / active_rival are NOT stored in session_state —
            # they are read from the engine via _es() on every render.
            "historial": [],
            # BattleEngine is the Game Logic Layer simulator.
            # It has the same interface as PokemonEnv's live-battle mode
            # but adds full battle mechanics (status, weather, hazards).
            # PokemonEnv is used ONLY in training scripts — never here.
            "env": BattleEngine(),
            "loaded_model": None,
            "current_model_path": "",
            "auto_enabled": False,
            # Turn counter shown in battle log
            "turn_number": 0,
            # When True in challenge mode, player must pick their next Pokémon
            "must_switch_rival": False,
            # Tracks the selected battle mode ("1. Simulación" / "2. Desafío")
            "battle_mode": "1. Simulación",
            # Moveset strategy selector ("competitive" / "balanced" / "random" / "custom")
            "moveset_mode": "competitive",
            # Per-Pokémon custom move overrides: {f"{prefix}_{idx}": [move_dict, …]}
            "custom_moves": {},
        }
    )


def predict_action_compatible(model, env):
    """
    Get the AI action for the current battle state.

    The *env* argument is now a BattleEngine instance (Game Logic Layer).
    BattleEngine._get_obs() delegates to obs_builder.build_obs_28(), which
    produces the IDENTICAL 28-dim vector that PokemonEnv._get_obs() would
    produce — so PPO models trained on PokemonEnv remain fully compatible.

    Decision pipeline
    -----------------
    1. If no model: pure greedy fallback (always valid, never immune moves).
    2. If model loaded: PPO.predict(obs) → raw_action.
    3. ai_advisor.get_hybrid_action() post-filters the raw action:
       - Blocks 0× (immune) move choices.
       - Overrides redundant status moves.
       - Prefers super-effective moves when they are clearly dominant.
       This filter is UI-only and does NOT affect PPO training.
    """
    ia_pokemon    = env.ia_pokemon
    rival_pokemon = env.rival_pokemon

    if model is None:
        if ia_pokemon and rival_pokemon:
            return get_greedy_action(ia_pokemon, rival_pokemon)
        raise RuntimeError("No PPO model loaded and no active Pokémon for greedy fallback.")

    # BattleEngine._get_obs() → obs_builder.build_obs_28() → 28-dim float32
    obs        = env._get_obs()
    ppo_action = int(model.predict(obs, deterministic=True)[0])

    if ia_pokemon and rival_pokemon:
        return get_hybrid_action(ppo_action, ia_pokemon, rival_pokemon)
    return ppo_action


@st.cache_data(show_spinner=False)
def _extract_model_step_count(filename: str) -> int:
    """
    Extract the training step count from a model zip filename.

    Handles the canonical checkpoint format:
        ppo_ckpt_500000_steps.zip  →  500000
        ppo_ckpt_1000000_steps.zip →  1000000

    Falls back to the last integer found in the name, or 0 if none.
    This ensures oldest (noob) models sort first and latest (professional)
    models sort last in the UI selector.
    """
    import re as _re
    m = _re.search(r'(\d+)[_\-]steps', filename)
    if m:
        return int(m.group(1))
    # Secondary heuristic: last numeric run in the stem
    nums = _re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else 0


def get_compatible_model_catalog(models_dir):
    compatible = []
    incompatible = []
    for root, _, files in os.walk(models_dir):
        for file_name in files:
            if not file_name.endswith(".zip"):
                continue
            relative_zip = os.path.relpath(os.path.join(root, file_name), models_dir)
            model_base_rel = relative_zip[:-4]
            model_base = os.path.join(models_dir, model_base_rel)
            compat = check_model_compatibility(model_base)
            if compat.is_valid:
                compatible.append(relative_zip)
            else:
                incompatible.append((relative_zip, compat.reason))
    # Sort compatible models oldest→newest by step count so the selector
    # shows the most "noob" checkpoint first and the most trained last.
    compatible_sorted = sorted(compatible, key=_extract_model_step_count)
    return compatible_sorted, sorted(incompatible)


def combat_step(action_ia, action_rival=None):
    """
    Execute one battle turn.

    Reads pre-turn state from the engine, calls engine.step(), then syncs
    session_state for rendering.  No direct team-dict mutation happens here —
    the engine is the sole mutator.
    """
    engine    = st.session_state.env
    # Read names from engine properties (already configured via _load_teams)
    curr_ia    = engine.ia_pokemon
    curr_rival = engine.rival_pokemon
    old_hp_rival = engine.hp_rival
    old_hp_ia    = engine.hp_ia

    _, _, _, _, info = engine.step(action_ia, action_rival=action_rival)

    st.session_state.turn_number = st.session_state.get("turn_number", 0) + 1
    turn_label = f"**T{st.session_state.turn_number}**"

    damage_to_rival = max(0, (old_hp_rival - engine.hp_rival) * 100)
    damage_to_ia    = max(0, (old_hp_ia    - engine.hp_ia)    * 100)

    ia_eff    = info.get("ia_effectiveness", "")
    rival_eff = info.get("rival_effectiveness", "")
    eff_tag_ia    = f" `{ia_eff}`"    if ia_eff    and ia_eff    not in ("Neutral", "") else ""
    eff_tag_rival = f" `{rival_eff}`" if rival_eff and rival_eff not in ("Neutral", "") else ""

    st.session_state.historial.insert(
        0,
        f"{turn_label} 🔴 **{curr_rival['name']}**: −{damage_to_ia:.1f}% HP | {info['rival_move']}{eff_tag_rival}",
    )
    st.session_state.historial.insert(
        0,
        f"{turn_label} ⚔️ **{curr_ia['name']}** (IA): −{damage_to_rival:.1f}% HP | {info['ia_move']}{eff_tag_ia}",
    )

    # ── Animation data for the arena (typewriter + blink + floating damage) ──
    # The engine log format is: "PokémonName used MoveName (Type) [Effectiveness]"
    # We extract just the MoveName for the dialog box.
    def _parse_move_name(log: str) -> str:
        m = _re.search(r' used (.+?) \(', log)
        return m.group(1) if m else ""

    _ia_log          = info.get("ia_move", "")
    _rival_log       = info.get("rival_move", "")
    _ia_move_name    = _parse_move_name(_ia_log)
    _rival_move_name = _parse_move_name(_rival_log)

    # Effectiveness labels for the dialog (omit Neutral to keep it clean)
    _eff_ia    = ia_eff    if ia_eff    and ia_eff    not in ("Neutral", "") else ""
    _eff_rival = rival_eff if rival_eff and rival_eff not in ("Neutral", "") else ""

    # Build separate message groups per attacker so the dialog can show
    # only the sub-turn of whoever acted last (no mixing of both sides).
    _msgs_ia = []
    if _ia_move_name:
        _msgs_ia.append(f"¡{curr_ia['name']} usó {_ia_move_name}!")
        if _eff_ia:
            _msgs_ia.append(f"¡{_eff_ia}!")
        if damage_to_rival > 0:
            _msgs_ia.append(f"{curr_rival['name']} recibió {damage_to_rival:.0f}% de daño.")

    _msgs_rival = []
    if _rival_move_name:
        _msgs_rival.append(f"¡{curr_rival['name']} usó {_rival_move_name}!")
        if _eff_rival:
            _msgs_rival.append(f"¡{_eff_rival}!")
        if damage_to_ia > 0:
            _msgs_rival.append(f"{curr_ia['name']} recibió {damage_to_ia:.0f}% de daño.")

    # "last_side": whoever attacked second in the turn shows their box last.
    # Rival messages are appended after IA in the log → rival = last when both moved.
    _last_side = "rival" if _msgs_rival else ("ia" if _msgs_ia else "ia")

    st.session_state["arena_anim"] = {
        "msgs_ia":         _msgs_ia,
        "msgs_rival":      _msgs_rival,
        "last_side":       _last_side,
        "damage_to_ia":    damage_to_ia,
        "damage_to_rival": damage_to_rival,
    }

    handle_post_turn_state()


def switch_rival_pokemon(new_index):
    """
    Voluntary rival switch in challenge mode.

    Delegates to engine.switch_turn() which handles the opponent's counter-
    attack and updates _active_rival / _rival_pokemon internally.  We then
    sync session_state from the engine and run post-turn faint checks.
    """
    engine = st.session_state.env
    state  = engine.get_state()

    if new_index == state["active_rival"]:
        return
    if state["team_rival"][new_index].get("debilitado"):
        return

    action_ia = predict_action_compatible(st.session_state.loaded_model, engine)

    # Pass the engine's internal team reference — switch_turn will use
    # list.index() to keep _active_rival in sync automatically.
    _, _, _, _, info = engine.switch_turn(
        side="rival",
        new_active_pokemon=state["team_rival"][new_index],
        opponent_action=action_ia,
    )

    damage_to_rival = max(0, info.get("hp_change_rival", 0.0) * 100)
    if info.get("switch_log"):
        st.session_state.historial.insert(
            0, f"🔁 **{engine.rival_pokemon['name']}**: {info['switch_log']}"
        )
    if info.get("ia_move") and info["ia_move"] != "No attack":
        st.session_state.historial.insert(
            0, f"⚔️ **{engine.ia_pokemon['name']}** (IA): -{damage_to_rival:.1f}% | {info['ia_move']}"
        )
    handle_post_turn_state()


def switch_ia_pokemon(new_index, action_rival=None):
    """
    Voluntary IA switch — called when the advisor decides a better-matchup
    Pokémon is available and the current matchup is clearly unfavourable.

    Delegates to engine.switch_turn("ia", ...) which:
      • Swaps _ia_pokemon / _active_ia in the engine.
      • Lets the rival attack during the switch turn (if action_rival is given).
      • Computes and logs the reward signal.

    Parameters
    ----------
    new_index    : int  — index in engine's team_ia list.
    action_rival : int | None — rival's move this turn (None = rival skips).
    """
    engine = st.session_state.env
    state  = engine.get_state()

    if new_index == state.get("active_ia", 0):
        return
    target = state["team_ia"][new_index]
    if target.get("debilitado", False):
        return

    old_name = engine.ia_pokemon["name"] if engine.ia_pokemon else "?"

    _, _, _, _, info = engine.switch_turn(
        side="ia",
        new_active_pokemon=target,
        opponent_action=action_rival,
    )

    damage_to_ia = max(0, info.get("hp_change_ia", 0.0) * 100)
    turn_label   = f"**T{st.session_state.get('turn_number', 0) + 1}**"
    st.session_state["turn_number"] = st.session_state.get("turn_number", 0) + 1

    st.session_state.historial.insert(
        0, f"{turn_label} 🔁 **IA** cambia: {old_name} → {engine.ia_pokemon['name']}"
    )
    if info.get("rival_move") and info["rival_move"] != "No attack":
        st.session_state.historial.insert(
            0, f"{turn_label} 🔴 **{engine.rival_pokemon['name']}**: -{damage_to_ia:.1f}% | {info['rival_move']}"
        )
    _sw_ia_msgs    = [f"¡IA retira a {old_name}!", f"¡Adelante, {engine.ia_pokemon['name']}!"]
    _sw_rival_msgs = []
    if info.get("rival_move") and damage_to_ia > 0:
        _sw_rival_move = _re.search(r' used (.+?) \(', info["rival_move"])
        _sw_rival_name = _sw_rival_move.group(1) if _sw_rival_move else info["rival_move"]
        _sw_rival_msgs.append(f"¡{engine.rival_pokemon['name']} usó {_sw_rival_name}!")
        _sw_rival_msgs.append(f"{engine.ia_pokemon['name']} recibió {damage_to_ia:.0f}% de daño.")
    _sw_last = "rival" if _sw_rival_msgs else "ia"
    st.session_state["arena_anim"] = {
        "msgs_ia":         _sw_ia_msgs,
        "msgs_rival":      _sw_rival_msgs,
        "last_side":       _sw_last,
        "damage_to_ia":    damage_to_ia,
        "damage_to_rival": 0,
    }
    handle_post_turn_state()


def get_switch_options(team, active_index):
    options = []
    for idx, pokemon in enumerate(team):
        if idx == active_index or pokemon["debilitado"]:
            continue
        options.append((idx, f"{pokemon['name']} ({int(pokemon['current_hp'] * 100)}% HP)"))
    return options


if not st.session_state.game_started:
    # ── Pokémon-themed background for the setup screen ────────────────────
    st.markdown(
        """
        <style>
        /* ── Main app background: vibrant Pokémon gradient ─────────────── */
        .stApp {
            background: linear-gradient(
                135deg,
                #1a1a2e 0%,
                #16213e 20%,
                #0f3460 40%,
                #1a1a5e 55%,
                #2d1b69 70%,
                #1a0a2e 85%,
                #0d1117 100%
            ) !important;
        }
        /* ── Floating Poké-ball decoration (CSS only, no images needed) ── */
        .stApp::before {
            content: "";
            position: fixed;
            top: -120px; right: -120px;
            width: 340px; height: 340px;
            border-radius: 50%;
            background: radial-gradient(circle at 40% 40%,
                #ff6b6b 0%, #ee5a24 50%, #c0392b 50.5%, #2c3e50 51%, #2c3e50 100%);
            opacity: 0.12;
            pointer-events: none;
            z-index: 0;
        }
        .stApp::after {
            content: "";
            position: fixed;
            bottom: -80px; left: -80px;
            width: 240px; height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle at 40% 40%,
                #74b9ff 0%, #0984e3 50%, #2c3e50 50.5%, #2c3e50 100%);
            opacity: 0.10;
            pointer-events: none;
            z-index: 0;
        }
        /* ── Cards / containers ─────────────────────────────────────────── */
        .stContainer, div[data-testid="stVerticalBlock"] {
            position: relative; z-index: 1;
        }
        /* ── Streamlit info / warning boxes ─────────────────────────────── */
        .stAlert {
            border-radius: 10px !important;
        }
        /* ── Divider line: electric yellow ──────────────────────────────── */
        hr {
            border-color: #f9ca24 !important;
            opacity: 0.35 !important;
        }
        /* ── Subheaders: golden Pokémon accent ───────────────────────────── */
        h2, h3 {
            color: #f9ca24 !important;
            text-shadow: 0 0 12px rgba(249,202,36,0.35);
        }
        /* ── Title ───────────────────────────────────────────────────────── */
        h1 {
            color: #ffffff !important;
            text-shadow: 0 2px 16px rgba(116, 185, 255, 0.5);
        }
        /* ── Radio + selectbox labels ────────────────────────────────────── */
        .stRadio label, .stSelectbox label {
            color: #dfe6e9 !important;
        }
        /* ── General text ────────────────────────────────────────────────── */
        p, li, .stMarkdown {
            color: #ecf0f1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚔️ Pokémon AI — Configuración de Equipos")
    pokemon_catalog = get_pokemon_catalog()
    item_catalog = get_item_catalog()
    st.caption("Los selectores son buscables: escribe para filtrar, navega con flechas y confirma con Enter.")

    # ── Moveset strategy selector ──────────────────────────────────────────
    st.divider()
    st.subheader("🧠 Estrategia de Moveset")
    st.markdown(
        """
        Elige cómo se construirán los **movimientos** de cada Pokémon antes del combate.
        El modo afecta a **ambos equipos** y se puede cambiar en cualquier momento
        antes de pulsar *Iniciar Combate*.

        | Modo | Descripción rápida |
        |---|---|
        | 🏆 **Competitive** | Prioriza STAB, cobertura de tipos y utilidad según el rol del Pokémon (el mejor para entrenar la IA). |
        | ⚖️ **Balanced** | Como Competitive pero permite hasta 2 movimientos del mismo tipo — mayor variedad. |
        | 🎲 **Random** | Los 4 primeros movimientos devueltos por la PokéAPI (comportamiento original). |
        | ✏️ **Custom** | Tú eliges manualmente desde el pool filtrado competitivo de cada Pokémon. |
        """
    )

    _MODE_LABELS = {
        "🏆 Competitive (Auto)": "competitive",
        "⚖️ Balanced":           "balanced",
        "🎲 Random (Legacy)":    "random",
        "✏️ Custom":             "custom",
    }
    _MODE_HELP = {
        "🏆 Competitive (Auto)": "Selección inteligente: STAB + cobertura + utilidad según el rol del Pokémon.",
        "⚖️ Balanced":           "Como Competitive pero permite hasta 2 movimientos del mismo tipo.",
        "🎲 Random (Legacy)":    "Los primeros 4 movimientos de la PokeAPI (comportamiento original).",
        "✏️ Custom":             "Elige manualmente desde el pool filtrado competitivo de cada Pokémon.",
    }
    _mode_col1, _mode_col2 = st.columns([2, 3])
    with _mode_col1:
        _selected_label = st.radio(
            "Modo de moveset:",
            list(_MODE_LABELS.keys()),
            key="moveset_mode_radio",
            help="Afecta a ambos equipos. Puedes cambiar el modo antes de iniciar la batalla.",
        )
    with _mode_col2:
        st.info(_MODE_HELP[_selected_label])

    moveset_mode: str = _MODE_LABELS[_selected_label]
    st.session_state.moveset_mode = moveset_mode

    # ── Moveset-mode change detection ─────────────────────────────────────────
    # get_pokemon_data(name, item) has NO moveset_mode in its cache key, so
    # switching moveset strategy NEVER re-fetches the Pokémon payload from the
    # PokeAPI.  Moves are built by generate_moveset() which reuses individually-
    # cached get_move_data() calls — also zero extra network traffic on a hit.
    #
    # What we DO clear on a mode switch:
    #   • custom_moves   — manual overrides must not bleed into other modes
    #   • ia/riv_slot_*  — Custom-expander selectbox values stay in session_state
    #                       even when the expander is collapsed; stale values here
    #                       would silently override the freshly computed moves
    # We do NOT touch ia_n_* / riv_n_* — the Pokémon roster is independent.
    _prev_moveset_mode = st.session_state.get("_prev_moveset_mode")
    if _prev_moveset_mode != moveset_mode:
        st.session_state.pop("custom_moves", None)
        for _mi in range(6):
            for _mpfx in ("ia", "riv"):
                for _slot in range(4):
                    st.session_state.pop(f"{_mpfx}_slot_{_mi}_{_slot}", None)
        st.session_state["_prev_moveset_mode"] = moveset_mode

    st.divider()

    # ── Team generation counter — forces full selectbox remount ────────────
    # Increment this any time a new team is written so that every
    # selectbox key changes and Streamlit creates fresh widgets with no
    # browser-cached state from the previous team.
    if "team_generation_id" not in st.session_state:
        st.session_state["team_generation_id"] = 0

    # ── Team rendering ─────────────────────────────────────────────────────
    # _apply_form_transforms() is defined at module level (above
    # `if "game_started" not in st.session_state`) and delegates form
    # resolution to src.pokemon_forms.resolve_form().  No local shadow needed.
    def render_team_selection(title: str, defaults: list, key_prefix: str, gen_id: int = 0):
        st.subheader(title)
        team = []
        cols = st.columns(3)

        for idx in range(6):
            with cols[idx % 3]:
                default_pokemon = (
                    defaults[idx] if defaults[idx] in pokemon_catalog else pokemon_catalog[0]
                )
                default_item = "Life Orb" if "Life Orb" in item_catalog else item_catalog[0]

                name = st.selectbox(
                    f"Pokémon {idx + 1}",
                    options=pokemon_catalog,
                    index=pokemon_catalog.index(default_pokemon),
                    key=f"{key_prefix}_n_{idx}_{gen_id}",
                    placeholder="Escribe para buscar Pokémon",
                )
                item = st.selectbox(
                    f"Objeto {idx + 1}",
                    options=item_catalog,
                    index=item_catalog.index(default_item),
                    key=f"{key_prefix}_i_{idx}_{gen_id}",
                    placeholder="Escribe para buscar objeto",
                )

                # ── Phase 1: base Pokémon data (cached per name+item only) ────
                # IMPORTANT: always materialise a fresh shallow copy before use.
                # @st.cache_data returns the same Python object on a cache hit
                # within a session.  If we kept a naked reference and later
                # mutated it (e.g. reset_pokemon_state), the cache entry would
                # be dirtied and every future hit would return corrupted data.
                # STRICT: get_pokemon_data has NO moveset_mode parameter — it
                # never re-fetches when the user switches moveset strategy.
                _cached = get_pokemon_data(name, item)
                # get_pokemon_data() NEVER returns None — it always falls back
                # to _build_fallback_pokemon().  This guard is belt-and-suspenders
                # for any unexpected future code path.
                if not _cached:
                    with st.container(border=True):
                        st.error(
                            f"❌ **{name}** no pudo cargarse.\n\n"
                            "Error inesperado en los datos. Elige otro Pokémon."
                        )
                    continue
                base_data = {**_cached}   # owned shallow copy — safe to modify

                # ── Phase 2: moveset generation (independent of base cache) ───
                # generate_moveset() builds moves from base_data["_all_move_names"]
                # using only individually-cached get_move_data() calls.
                # Changing moveset_mode costs zero extra PokeAPI Pokémon fetches.
                moves, move_pool = generate_moveset(base_data, moveset_mode)

                # Apply custom move overrides (only in custom mode)
                custom_key = f"{key_prefix}_{idx}"
                if moveset_mode == "custom":
                    override = st.session_state.get("custom_moves", {}).get(custom_key)
                    if override and len(override) == 4:
                        moves = override

                data = {**base_data, "moves": moves, "move_pool": move_pool, "held_item_name": item}

                # ── Phase 3: form post-processing ─────────────────────────────
                # NOTE: Mega Evolution is intentionally NOT applied here.
                # Team selection always shows the BASE form — Mega Evolution is
                # a player decision taken during battle (via the MEGA button).
                # GMax / Dynamax are also battle mechanics, not team-setup state.
                # The "held_item_name" key carries the item name so BattleEngine
                # and the battle UI know which stone is held without transforming yet.

                with st.container(border=True):
                    # Sprite + item
                    sprite_col, info_col = st.columns([1, 2])
                    sprite_col.image(_safe_sprite(data, "front"), width=68)
                    with info_col:
                        st.markdown(f"**{data['name']}**")
                        # Type badges
                        badges_html = "".join(
                            type_badge_html(t) for t in data["types"]
                        )
                        st.markdown(badges_html, unsafe_allow_html=True)
                        # Role badge
                        ri = data.get("role_info", {})
                        if ri:
                            role_html = (
                                f'<span style="font-size:10px;font-weight:bold;'
                                f'color:{ri.get("color","#888")};">'
                                f'{ri.get("label","")}</span>'
                            )
                            st.markdown(role_html, unsafe_allow_html=True)
                        if data.get("item"):
                            st.image(data["item"]["sprite"], width=28,
                                     caption=data["item"]["name"])
                        # Shiny indicator — shown when shiny flag is set
                        if data.get("shiny"):
                            st.caption("✨ ¡Shiny!")
                        # Offline indicator — shown when API was unreachable
                        if data.get("_is_fallback"):
                            st.caption("📡 datos offline")

                    st.divider()
                    # Move list with type badge + power
                    for move in data["moves"]:
                        mtype   = move.get("type", "normal")
                        mpower  = move.get("power") or 0
                        mdc     = move.get("damage_class", "status")
                        badge   = type_badge_html(mtype, small=True)
                        pwr_str = f"Pwr {mpower}" if mpower else "Status"
                        cls_icon = "⚔️" if mdc == "physical" else ("✨" if mdc == "special" else "🔮")
                        st.markdown(
                            f"{badge} {cls_icon} **{move['name']}** — {pwr_str}",
                            unsafe_allow_html=True,
                        )

                    # ── Custom editing expander ────────────────────────────
                    if moveset_mode == "custom" and base_data.get("move_pool"):
                        pool = base_data["move_pool"]
                        option_names  = [m["name"] for m in pool]
                        option_by_name = {m["name"]: m for m in pool}

                        with st.expander("✏️ Personalizar movimientos"):
                            current_moves = data["moves"]
                            new_custom: list[dict] = []

                            for slot in range(4):
                                default_name = (
                                    current_moves[slot]["name"]
                                    if slot < len(current_moves)
                                    else option_names[0]
                                )
                                safe_idx = (
                                    option_names.index(default_name)
                                    if default_name in option_names
                                    else 0
                                )
                                sel_name = st.selectbox(
                                    f"Slot {slot + 1}",
                                    options=option_names,
                                    index=safe_idx,
                                    key=f"{key_prefix}_slot_{idx}_{slot}",
                                )
                                new_custom.append(
                                    option_by_name.get(sel_name, pool[0])
                                )

                            # Persist selection in session state
                            if "custom_moves" not in st.session_state:
                                st.session_state.custom_moves = {}
                            st.session_state.custom_moves[custom_key] = new_custom
                            # Reflect changes in the card immediately
                            data = {**data, "moves": new_custom}

                team.append(data)
        return team

    # ── Team preset system ────────────────────────────────────────────────────
    # Curated competitive teams.  Each entry is (ia_team, rival_team) with 6
    # Pokémon names apiece.  Names must exist in the PokeAPI catalog.
    COMPETITIVE_PRESETS: dict[str, dict] = {
        "🏆 Balanced Offensive": {
            "description": "Versatile all-rounder team balancing physical, special, and utility.",
            "ia":    ["Mewtwo",    "Garchomp",  "Starmie",  "Scizor",    "Heatran",   "Togekiss"],
            "rival": ["Dragonite", "Tyranitar", "Gengar",   "Gyarados",  "Salamence", "Lucario"],
        },
        "🐉 Dragon Slayer": {
            "description": "Ice + Fairy + Steel coverage to dismantle Dragon-heavy teams.",
            "ia":    ["Mamoswine", "Clefable",  "Scizor",   "Weavile",   "Togekiss",  "Jolteon"],
            "rival": ["Rayquaza",  "Dragonite", "Garchomp", "Salamence", "Latios",    "Kingdra"],
        },
        "☀️ Sun Team": {
            "description": "Drought-powered team: Solar Beam, Fire moves boosted, Chlorophyll sweepers.",
            "ia":    ["Charizard", "Venusaur",  "Ninetales", "Heatran",  "Victreebel","Arcanine"],
            "rival": ["Kyogre",    "Blastoise", "Vaporeon",  "Politoed", "Ludicolo",  "Swampert"],
        },
        "🌊 Rain Squad": {
            "description": "Swift Swim sweepers backed by Drizzle rain support.",
            "ia":    ["Kyogre",    "Kabutops",  "Ludicolo",  "Toxicroak","Kingdra",   "Starmie"],
            "rival": ["Groudon",   "Charizard", "Ninetales", "Heatran",  "Volcarona", "Arcanine"],
        },
        "👻 Ghost & Psychic": {
            "description": "Trick Room + Psychic/Ghost types for mind-game dominance.",
            "ia":    ["Mewtwo",    "Gengar",    "Alakazam",  "Slowbro",  "Chandelure","Espeon"],
            "rival": ["Tyranitar", "Krookodile","Bisharp",   "Incineroar","Pangoro",  "Absol"],
        },
    }

    # ── Team selection mode ───────────────────────────────────────────────────
    st.divider()
    st.subheader("👥 Selección de Equipos")
    st.markdown(
        """
        Elige cómo se forman los **equipos** que entrarán al combate.
        Ambos equipos (IA y rival) se configuran en el mismo modo.

        | Modo | Descripción rápida |
        |---|---|
        | 🎲 **Random** | 6 Pokémon base elegidos al azar del catálogo — ideal para entrenamientos variados. |
        | ✏️ **Custom** | Selecciona manualmente cada Pokémon y su objeto — control total. |
        | 🏆 **Competitive Preset** | Equipos curados inspirados en el meta competitivo (VGC / Smogon) — listo para combatir. |
        """
    )
    _team_mode_options = ["🎲 Random", "✏️ Custom", "🏆 Competitive Preset"]
    _team_mode = st.radio(
        "Modo de equipo:",
        _team_mode_options,
        index=0,
        key="team_selection_mode",
        horizontal=True,
        help=(
            "Random: equipos por defecto variados. "
            "Custom: elige tus propios Pokémon. "
            "Competitive Preset: equipos curados listos para competir."
        ),
    )

    # ── Helper: pick 6 unique random BASE Pokémon from the catalog ───────────
    def _pick_random_team() -> list[str]:
        # is_base_pokemon() is applied here as a third-layer safety net —
        # the catalog is already filtered, but this prevents any stale
        # session_state value from contaminating a newly generated team.
        valid = [p for p in pokemon_catalog if is_base_pokemon(p)]
        pool  = valid if len(valid) >= 6 else (valid * 6)
        return random.sample(pool, 6)

    # ── STEP 1 ─ Render conditional widgets so their values are known ──────────
    # The preset selectbox must render (and store its value) BEFORE we do
    # mode-change detection, because the detection needs _cur_preset.
    _cur_preset = ""
    if _team_mode == "🏆 Competitive Preset":
        _preset_names = list(COMPETITIVE_PRESETS.keys())
        _chosen_preset_label = st.selectbox(
            "Elige un preset competitivo:",
            _preset_names,
            key="competitive_preset_choice",
        )
        _cur_preset = _chosen_preset_label

    # ── STEP 2 ─ Mode-change detection (runs with full info) ──────────────────
    # We now know both _team_mode and _cur_preset.  Only here can we correctly
    # decide whether anything changed and what to regenerate.
    #
    # Critical insight: this block must run BEFORE we set _default_ia/_default_rival.
    # If we set defaults first and detect changes second, any session_state writes
    # (e.g. rand_team_ia) made in step 2 would be invisible to the defaults that
    # were already assigned — the classic ordering race condition.
    _prev_mode   = st.session_state.get("_prev_team_mode")
    _prev_preset = st.session_state.get("_prev_preset_choice", "")

    _mode_changed   = _prev_mode != _team_mode
    _preset_changed = _team_mode == "🏆 Competitive Preset" and _prev_preset != _cur_preset

    if _mode_changed or _preset_changed:
        # ── Why we need st.rerun() here ──────────────────────────────────────
        # Streamlit widget state has TWO representations per widget key:
        #
        #   1. Server-side  — st.session_state["ia_n_0"] = "Charizard"
        #   2. Browser-side — the widget value bundled into the HTTP request
        #                      that triggered this render
        #
        # The reconciler runs INSIDE the same render pass as this code.
        # Pop order:
        #   a) We call st.session_state.pop("ia_n_0") → removes server value
        #   b) Streamlit reconciler runs st.selectbox(key="ia_n_0", index=N)
        #   c) The browser-sent value for "ia_n_0" ("Charizard") is present
        #      in the request → the reconciler writes it BACK into session_state
        #   d) The selectbox ignores index= and renders "Charizard"
        #
        # Consequence: the card BODY (name → get_pokemon_data → sprite) also
        # gets "Charizard" because it reads the selectbox return value.
        # Everything is self-consistent but frozen at the OLD team.
        #
        # The ONLY reliable fix: write new state, clear keys, then st.rerun().
        # The NEXT render starts with a fresh browser request that carries NO
        # values for the cleared keys → index= is authoritative → correct team.
        #
        # Guard: skip the rerun on the very first load (_prev_mode is None).
        # On first load no widgets have been rendered yet, so there are no
        # browser-cached values to flush.  An unconditional rerun would simply
        # double the initial load time with zero benefit.
        # ─────────────────────────────────────────────────────────────────────

        # Wipe all Pokémon-name and item selectbox keys so that the index= param
        # in render_team_selection() is honoured on the next render.
        for _i in range(6):
            for _pfx in ("ia", "riv"):
                st.session_state.pop(f"{_pfx}_n_{_i}", None)
                st.session_state.pop(f"{_pfx}_i_{_i}", None)
        st.session_state.pop("custom_moves", None)

        # For Random: generate the new team NOW (selectbox keys were just cleared,
        # so the upcoming rerun will render them via index=).
        # For other modes: discard any previously stored random team.
        if _team_mode == "🎲 Random":
            st.session_state["rand_team_ia"]    = _pick_random_team()
            st.session_state["rand_team_rival"] = _pick_random_team()
        else:
            st.session_state.pop("rand_team_ia",    None)
            st.session_state.pop("rand_team_rival", None)

        st.session_state["_prev_team_mode"]     = _team_mode
        st.session_state["_prev_preset_choice"] = _cur_preset

        # Bump generation counter — all selectbox keys change on the next render,
        # guaranteeing no browser-cached widget values survive the mode switch.
        st.session_state["team_generation_id"] = st.session_state.get("team_generation_id", 0) + 1

        # Flush browser-cached widget state.  The next render sees clean keys
        # and honours index= for every Pokémon / item selectbox.
        if _prev_mode is not None:          # not first load — rerun needed
            st.rerun()

    # ── STEP 3 ─ Resolve defaults (reads session_state written in step 2) ──────
    _HARDCODED_IA    = ["Mewtwo",    "Rayquaza",  "Kyogre",   "Groudon",   "Metagross", "Sceptile"]
    _HARDCODED_RIVAL = ["Charizard", "Blastoise", "Venusaur", "Gengar",    "Lucario",   "Tyranitar"]

    _default_ia:    list[str] = _HARDCODED_IA
    _default_rival: list[str] = _HARDCODED_RIVAL

    if _team_mode == "🏆 Competitive Preset":
        _preset = COMPETITIVE_PRESETS[_cur_preset]
        st.caption(f"📋 {_preset['description']}")
        _default_ia    = _preset["ia"]
        _default_rival = _preset["rival"]

    elif _team_mode == "🎲 Random":
        # rand_team_ia was set in step 2 (on mode entry) or persists from the
        # previous render.  Either way it is always valid here.
        _default_ia    = st.session_state.get("rand_team_ia",    _HARDCODED_IA)
        _default_rival = st.session_state.get("rand_team_rival", _HARDCODED_RIVAL)

        _rc1, _rc2 = st.columns([3, 1])
        with _rc1:
            st.caption("Equipos generados aleatoriamente. Pulsa el botón para obtener nuevos Pokémon.")
        with _rc2:
            if st.button("🎲 Regenerar", key="btn_regen_random", use_container_width=True):
                # ── Why st.rerun() is required here ──────────────────────────
                # Streamlit widgets hold TWO representations of their value:
                #   1. The Python session_state dict (server-side)
                #   2. The browser-side widget state sent with each interaction
                #
                # When a button is clicked, the browser sends the current widget
                # values along with the click event.  Even if we pop "ia_n_0"
                # from session_state and then call st.selectbox(key="ia_n_0",
                # index=N), Streamlit's reconciler may still restore the browser-
                # sent value for that key, overriding the new index=.
                #
                # The only reliable fix: write new teams to session_state, clear
                # widget keys, then call st.rerun().  The subsequent render starts
                # with NO browser-cached values for the cleared keys, so index=
                # is honoured and the selectboxes show the new Pokémon.
                st.session_state["rand_team_ia"]    = _pick_random_team()
                st.session_state["rand_team_rival"] = _pick_random_team()
                for _ri in range(6):
                    for _rpfx in ("ia", "riv"):
                        st.session_state.pop(f"{_rpfx}_n_{_ri}", None)
                        st.session_state.pop(f"{_rpfx}_i_{_ri}", None)
                st.session_state.pop("custom_moves", None)
                st.session_state["team_generation_id"] = st.session_state.get("team_generation_id", 0) + 1
                st.rerun()  # start a clean render — selectboxes pick up new index=

    elif _team_mode == "✏️ Custom":
        st.caption("Elige tus Pokémon manualmente. Los movimientos no se sobreescriben al cambiar estrategia.")

    col_ia, col_rival = st.columns(2)
    _gen_id = st.session_state.get("team_generation_id", 0)
    with col_ia:
        team_ia = render_team_selection(
            "🤖 Equipo IA",
            _default_ia,
            "ia",
            gen_id=_gen_id,
        )
    with col_rival:
        team_rival = render_team_selection(
            "👤 Equipo Rival",
            _default_rival,
            "riv",
            gen_id=_gen_id,
        )

    if st.button("🔥 INICIAR COMBATE", type="primary", use_container_width=True):
        if len(team_ia) == 6 and len(team_rival) == 6:
            try:
                conn = sqlite3.connect("pokemon_bigdata.db")
                conn.execute("DROP TABLE IF EXISTS v_logs")
                conn.execute(
                    """
                    CREATE TABLE v_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ia_move_name TEXT,
                        rival_move TEXT,
                        ia_move_type TEXT,
                        rival_move_type TEXT,
                        ia_effectiveness TEXT,
                        rival_effectiveness TEXT,
                        hp_ia REAL,
                        hp_rival REAL,
                        reward REAL
                    )
                    """
                )
                conn.close()
            except Exception:
                pass

            # Hand the configured teams to a fresh BattleEngine.
            # The engine deep-copies both lists and resets all Pokémon to full
            # battle-start state internally — no external reset loop needed.
            new_engine = BattleEngine(team_ia, team_rival, log_to_db=True)
            st.session_state.update({
                "env":             new_engine,
                "game_started":    True,
                "battle_finished": False,
                "resultado":       "",
                "historial":       [],
                "turn_number":     0,
                "must_switch_rival": False,
                # Mega Evolution tracking — one mega per side per battle
                "rival_mega_used":      False,
                "rival_mega_confirmed": False,
                "ia_mega_used":         False,
                # active_ia / active_rival are NOT stored here — read from engine
            })
            # Engine owns all team data. UI reads via _es() on every render.
            st.rerun()
        else:
            st.error("Asegúrate de que todos los Pokémon hayan cargado correctamente.")
    st.stop()


# ── Global CSS for the battle screen ─────────────────────────────────────
# Injected via st.markdown so it affects the Streamlit parent page
# (not the st.html() iframes, which are sandboxed separately).
st.markdown(
    """
    <style>
    /* 1. Fondo global — gradiente profundo que cubre toda la app */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }

    /* 2. Contenedores transparentes para no bloquear el fondo */
    .block-container,
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    .stMarkdown,
    .element-container,
    .stColumn {
        background: transparent !important;
    }

    /* 3. Sidebar — fondo oscuro semi-transparente para que resalte */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85) !important;
        backdrop-filter: blur(6px);
    }

    /* 4. Tipografía general en blanco sobre fondo oscuro */
    p, li, label, .stMarkdown {
        color: #ecf0f1 !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Read authoritative battle state from engine ONCE per render cycle.
# All rendering code below uses these local variables — no session_state
# team mirrors exist; desync is structurally impossible.
_state        = _es()
_team_ia      = _state["team_ia"]
_team_rival   = _state["team_rival"]
_active_ia    = _state["active_ia"]
_active_rival = _state["active_rival"]

with st.sidebar:
    st.title("🕹️ Panel de Control")
    mode = st.radio("Modo:", ["1. Simulación", "2. Desafío"], key="battle_mode")

    model_list, incompatible_models = get_compatible_model_catalog(MODELS_DIR)
    if model_list:
        selected_model = st.selectbox("Modelo PPO compatible:", model_list)
        model_path = os.path.join(MODELS_DIR, selected_model)
        if st.session_state.current_model_path != model_path:
            model_base = model_path[:-4]
            st.session_state.loaded_model = require_compatible_model(model_base)
            st.session_state.current_model_path = model_path
            st.success(f"Cerebro cargado: {selected_model}")
    else:
        st.error("No compatible PPO models were found. Train a new model with the canonical environment first.")
    if incompatible_models:
        with st.expander("Modelos bloqueados (LEGACY - INCOMPATIBLE)"):
            for model_name, reason in incompatible_models:
                st.caption(f"{model_name}: {reason}")

    auto = st.toggle("Auto-Play", value=st.session_state.auto_enabled, disabled=st.session_state.loaded_model is None)
    st.session_state.auto_enabled = auto
    speed = st.slider("Velocidad", 0.1, 2.0, 0.5)

    current_ia = _team_ia[_active_ia]
    st.divider()
    # Show active Pokémon role badge if available
    ri = current_ia.get("role_info", {})
    if ri:
        st.markdown(
            f'<span style="font-size:11px;font-weight:bold;color:{ri.get("color","#888")};">'
            f'{ri.get("label","")} — {ri.get("desc","")}</span>',
            unsafe_allow_html=True,
        )
    # Show moveset mode
    _mode_label = {
        "competitive": "🏆 Competitive", "balanced": "⚖️ Balanced",
        "random": "🎲 Random", "custom": "✏️ Custom",
    }.get(st.session_state.get("moveset_mode", "competitive"), "")
    if _mode_label:
        st.caption(f"Moveset: {_mode_label}")
    st.subheader(f"📊 Stats: {current_ia['name']}")
    st.table(pd.Series(current_ia["stats"]))
    st.caption(f"Tipos: {' / '.join(format_name(type_name) for type_name in current_ia['types'])}")
    st.caption(f"HP actual: {int(current_ia['current_hp'] * 100)}%")

    with st.expander("Tabla de tipos"):
        st.dataframe(pd.DataFrame(build_type_chart_rows()), use_container_width=True, hide_index=True)

if st.session_state.loaded_model is None:
    st.error("No compatible PPO model is loaded. Battle execution is blocked.")
    st.stop()


current_ia    = _team_ia[_active_ia]
current_rival = _team_rival[_active_rival]

# ── Mega display preview ──────────────────────────────────────────────────────
# If the player has toggled "MEGA EVOLUCIONAR" but hasn't attacked yet,
# apply the form transform to a DISPLAY-ONLY copy so the arena shows the
# mega sprite, name and types immediately — without touching the engine.
# The real engine update happens inside the attack button handler below.
_rival_item_name_pre = (current_rival.get("item") or {}).get("name", "")
_rival_item_key_pre  = _rival_item_name_pre.lower().strip().replace(" ", "-")
if (
    st.session_state.get("rival_mega_confirmed", False)
    and not st.session_state.get("rival_mega_used", False)
    and not current_rival.get("mega_evolved", False)
    and _rival_item_key_pre in MEGA_STONE_MAP
):
    current_rival = _apply_form_transforms(dict(current_rival), _rival_item_name_pre)

# Pre-compute values used in the arena HTML
_hp_ia    = float(st.session_state.env.hp_ia)
_hp_rival = float(st.session_state.env.hp_rival)
_bar_ia    = hp_bar_color(_hp_ia)
_bar_rival = hp_bar_color(_hp_rival)
_types_ia_html    = "".join(type_badge_html(t) for t in current_ia["types"])
_types_rival_html = "".join(type_badge_html(t) for t in current_rival["types"])
_status_ia_html    = status_badge_html(current_ia.get("status"))
_status_rival_html = status_badge_html(current_rival.get("status"))
_weather_html = weather_badge_html(st.session_state.env.weather)
_turn_html = (
    f'<div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);'
    f'background:rgba(0,0,0,0.7);color:#fff;padding:3px 12px;border-radius:8px;'
    f'font-size:13px;">Turno {st.session_state.get("turn_number", 0)}'
    f"{_weather_html}</div>"
    if st.session_state.get("turn_number", 0) > 0 else ""
)

# ── Arena layout: perspective depends on mode ────────────────────────────────
# Simulación: IA (back/left) vs Rival (front/right)  — observing the IA play
# Desafío:    Tú (back/left) vs IA (front/right)     — you face the IA
_challenge_mode = st.session_state.get("battle_mode", "1. Simulación") == "2. Desafío"

# ── Animation data ────────────────────────────────────────────────────────────
_anim          = st.session_state.get("arena_anim", {})
_anim_msgs_ia  = _anim.get("msgs_ia",    [])
_anim_msgs_riv = _anim.get("msgs_rival", [])
_last_side     = _anim.get("last_side",  "")
_dmg_to_ia     = _anim.get("damage_to_ia",    0)
_dmg_to_rival  = _anim.get("damage_to_rival", 0)

# Map damage → left/right visual side
if _challenge_mode:
    _left_dmg  = _dmg_to_rival   # human (left) took damage from IA
    _right_dmg = _dmg_to_ia      # IA (right) took damage from human
else:
    _left_dmg  = _dmg_to_ia      # IA (left) took damage from rival
    _right_dmg = _dmg_to_rival   # rival (right) took damage from IA

_left_blink  = "id=\"sprite-left\"  class=\"hit-blink\"" if _left_dmg  > 0 else "id=\"sprite-left\""
_right_blink = "id=\"sprite-right\" class=\"hit-blink\"" if _right_dmg > 0 else "id=\"sprite-right\""
_left_float  = (f'<div class="dmg-float" style="bottom:185px;left:90px;">-{_left_dmg:.0f}%</div>'
                if _left_dmg  > 0 else "")
_right_float = (f'<div class="dmg-float" style="top:85px;right:90px;">-{_right_dmg:.0f}%</div>'
                if _right_dmg > 0 else "")

# ── Dialog: construir un cuadro por atacante, ambos visibles en el turno ─
# Clase CSS según posición en pantalla (depende del modo):
#   Sim:     IA = izquierda → dialog-player  |  Rival = derecha → dialog-enemy
#   Desafío: IA = derecha   → dialog-enemy   |  Rival = izquierda → dialog-player
_ia_css    = "dialog-player" if not _challenge_mode else "dialog-enemy"
_rival_css = "dialog-enemy"  if not _challenge_mode else "dialog-player"

def _build_dialog_box(msgs, css_class, delay_offset=0.0):
    """Devuelve el HTML de un cuadro de diálogo estilo Game Boy para msgs."""
    if not msgs:
        return ""
    lines = "".join(
        f'<div style="animation:dlgLineFade 0.4s ease-out {delay_offset + i*0.15:.2f}s both;">'
        f'{m.upper()}</div>'
        for i, m in enumerate(msgs[-2:])   # máx 2 líneas por cuadro
    )
    cursor = '<span class="gb-cursor">&#9608;</span>'
    return (
        f'<div style="display:flex;width:100%;margin-top:18px;">'
        f'  <div class="gb-dialog {css_class}">'
        f'    <div class="gb-text">{lines}{cursor}</div>'
        f'  </div>'
        f'</div>'
    )

# Cuadro de la IA (si atacó) + cuadro del rival (si atacó), en orden de turno
_box_ia    = _build_dialog_box(_anim_msgs_ia,  _ia_css,    delay_offset=0.0)
_box_rival = _build_dialog_box(_anim_msgs_riv, _rival_css, delay_offset=0.25)

if _box_ia or _box_rival:
    _all_dialog_boxes = _box_ia + _box_rival
else:
    # Estado inicial — ningún turno jugado aún
    _all_dialog_boxes = (
        '<div style="display:flex;width:100%;margin-top:18px;">'
        '  <div class="gb-dialog dialog-player">'
        '    <div class="gb-text" style="color:#8899bb;">— ELIGE UN MOVIMIENTO —</div>'
        '  </div>'
        '</div>'
    )

if _challenge_mode:
    # Left (bottom) = human player (rival) shown from behind
    # Right (top)   = IA opponent shown from front
    _left_pokemon      = current_rival
    _left_label        = current_rival["name"]         # "Tú"
    _left_hp           = _hp_rival
    _left_bar          = _bar_rival
    _left_types_html   = _types_rival_html
    _left_status_html  = _status_rival_html
    _left_sprite_src   = _path_to_data_uri(_safe_sprite(current_rival, "back"))
    _left_border_color = "#00d4ff"

    _right_pokemon     = current_ia
    _right_label       = f"{current_ia['name']} (IA)"
    _right_hp          = _hp_ia
    _right_bar         = _bar_ia
    _right_types_html  = _types_ia_html
    _right_status_html = _status_ia_html
    _right_sprite_src  = _path_to_data_uri(_safe_sprite(current_ia, "front"))
    _right_border_color= "#ff4b4b"
else:
    # Left (bottom) = IA shown from behind
    # Right (top)   = Rival shown from front
    _left_pokemon      = current_ia
    _left_label        = f"{current_ia['name']} (IA)"
    _left_hp           = _hp_ia
    _left_bar          = _bar_ia
    _left_types_html   = _types_ia_html
    _left_status_html  = _status_ia_html
    _left_sprite_src   = _path_to_data_uri(_safe_sprite(current_ia, "back"))
    _left_border_color = "#00d4ff"

    _right_pokemon     = current_rival
    _right_label       = current_rival["name"]
    _right_hp          = _hp_rival
    _right_bar         = _bar_rival
    _right_types_html  = _types_rival_html
    _right_status_html = _status_rival_html
    _right_sprite_src  = _path_to_data_uri(_safe_sprite(current_rival, "front"))
    _right_border_color= "#ff4b4b"

st.html(
    f"""
    <style>
      /* ── CAMBIO 1: @import AL PRINCIPIO, antes de cualquier regla ──────── */
      @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

      /* ── Animations ──────────────────────────────────────────────────── */
      @keyframes hitBlink {{
        0%,100% {{ opacity:1; }}
        20%,60% {{ opacity:0.05; }}
        40%,80% {{ opacity:1; }}
      }}
      @keyframes floatUp {{
        0%   {{ opacity:1; transform:translateY(0); }}
        100% {{ opacity:0; transform:translateY(-38px); }}
      }}
      /* cuadro completo: fade-in + deslizamiento suave al aparecer */
      @keyframes dlgBoxFade {{
        from {{ opacity:0; transform:translateY(6px); }}
        to   {{ opacity:1; transform:translateY(0);   }}
      }}
      /* líneas individuales dentro del cuadro */
      @keyframes dlgLineFade {{
        from {{ opacity:0; }}
        to   {{ opacity:1; }}
      }}
      @keyframes gbBlink {{
        0%, 49% {{ opacity:1; }}
        50%,100% {{ opacity:0; }}
      }}

      /* ── Combat sprites ──────────────────────────────────────────────── */
      .hit-blink {{ animation: hitBlink 0.18s ease-in-out 4; }}

      /* ── Floating damage chip ────────────────────────────────────────── */
      .dmg-float {{
        position:absolute; background:rgba(220,30,30,0.92);
        color:#fff; font-weight:bold; font-size:13px;
        padding:3px 9px; border-radius:10px; pointer-events:none;
        animation: floatUp 1.4s ease-out forwards;
        z-index:20;
      }}

      /* ── HP bars ─────────────────────────────────────────────────────── */
      #hp-bar-left  {{ transition: width 0.55s ease-out; }}
      #hp-bar-right {{ transition: width 0.55s ease-out; }}

      /* ── Game Boy dialog box — base ─────────────────────────────────── */
      .gb-dialog {{
        position: relative;
        display: table;
        max-width: 450px;
        min-width: 220px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border: 2px solid #4e4eef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        padding: 14px 18px;
        min-height: 70px;
        height: auto;
        box-sizing: border-box;
        animation: dlgBoxFade 0.5s ease-out both;
      }}

      /* ── Alineación izquierda ────────────────────────────────────────── */
      .dialog-player {{ margin-right: auto; margin-left: 0; }}

      /* ── Alineación derecha ──────────────────────────────────────────── */
      .dialog-enemy  {{ margin-left: auto; margin-right: 0; }}

      /* ── Flecha – capa exterior: color del borde azul ───────────────── */
      .gb-dialog::before {{
        content: '';
        position: absolute;
        top: -14px;
        width: 0; height: 0;
        border-left: 9px solid transparent;
        border-right: 9px solid transparent;
        border-bottom: 14px solid #4e4eef;
      }}
      .dialog-player::before {{ left: 28px; }}
      .dialog-enemy::before  {{ right: 28px; left: auto; }}

      /* ── Flecha – capa interior: color inicio del gradiente ─────────── */
      .gb-dialog::after {{
        content: '';
        position: absolute;
        top: -9px;
        width: 0; height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-bottom: 10px solid #1a1a2e;
      }}
      .dialog-player::after {{ left: 30px; }}
      .dialog-enemy::after  {{ right: 30px; left: auto; }}

      /* ── Fuente pixelada ─────────────────────────────────────────────── */
      .gb-text {{
        font-family: 'Press Start 2P', cursive !important;
        font-size: 11px !important;
        line-height: 1.9;
        color: #ffffff !important;
        image-rendering: pixelated;
        -webkit-font-smoothing: none;
      }}
      .gb-cursor {{
        font-family: 'Press Start 2P', cursive !important;
        font-size: 11px;
        color: #ffffff;
        animation: gbBlink 0.6s step-end infinite;
        margin-left: 3px;
      }}
    </style>

    <!-- ── Arena ──────────────────────────────────────────────────────── -->
    <div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png');
         background-size: cover; height: 270px; border-radius: 20px 20px 0 0;
         position: relative; border: 3px solid #444;">
      {_turn_html}

      <!-- Floating damage -->
      {_left_float}
      {_right_float}

      <!-- Right side (top-right) — opponent -->
      <div style="position:absolute;top:22px;right:50px;width:245px;
                  background:rgba(0,0,0,0.82);padding:9px 12px;
                  border-radius:12px;color:white;border-left:5px solid {_right_border_color};">
        <b style="font-size:14px;">{_right_label}</b>
        {_right_status_html}
        <div style="margin:3px 0;">{_right_types_html}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
          <div style="flex:1;background:#333;height:9px;border-radius:5px;overflow:hidden;">
            <div id="hp-bar-right" style="width:{_right_hp*100:.1f}%;background:{_right_bar};
                        height:100%;border-radius:5px;"></div>
          </div>
          <span style="font-size:11px;min-width:34px;text-align:right;">{int(_right_hp*100)}%</span>
        </div>
        <img src="{_right_sprite_src}" {_right_blink}
             style="position:absolute;top:65px;right:8px;" width="90">
      </div>

      <!-- Left side (bottom-left) — player perspective -->
      <div style="position:absolute;bottom:22px;left:50px;width:245px;
                  background:rgba(0,0,0,0.82);padding:9px 12px;
                  border-radius:12px;color:white;border-left:5px solid {_left_border_color};">
        <b style="font-size:14px;">{_left_label}</b>
        {_left_status_html}
        <div style="margin:3px 0;">{_left_types_html}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
          <div style="flex:1;background:#333;height:9px;border-radius:5px;overflow:hidden;">
            <div id="hp-bar-left" style="width:{_left_hp*100:.1f}%;background:{_left_bar};
                        height:100%;border-radius:5px;"></div>
          </div>
          <span style="font-size:11px;min-width:34px;text-align:right;">{int(_left_hp*100)}%</span>
        </div>
        <img src="{_left_sprite_src}" {_left_blink}
             style="position:absolute;bottom:90px;left:8px;" width="108">
      </div>
    </div>

    <!-- ── Game Boy dialog boxes (uno por atacante) ─────────────────── -->
    {_all_dialog_boxes}
    """
)

# ── Team panels — retro pixel style ───────────────────────────────────────
def _make_team_panel(team, active_idx, border_color, label):
    """Build the HTML for one 6-sprite team panel with a tab label."""
    items = []
    for idx, poke in enumerate(team):
        uri  = _path_to_data_uri(_safe_sprite(poke, "front"))
        opa  = "1" if not poke["debilitado"] else "0.18"
        # Active Pokémon: glowing bottom border; fainted: greyscale filter
        bbot   = f"3px solid {border_color}" if idx == active_idx else "3px solid transparent"
        filt   = "grayscale(100%)" if poke["debilitado"] else "none"
        items.append(
            f'<div style="flex:1;text-align:center;padding:0 2px 4px 2px;border-bottom:{bbot};">'
            f'  <img src="{uri}" width="44"'
            f'       style="opacity:{opa};filter:{filt};display:block;margin:0 auto;">'
            f'</div>'
        )
    sprites_row = "".join(items)
    # Tab label positioned above the top border of the container
    tab = (
        f'<div style="'
        f'  position:absolute;top:-20px;left:0;'
        f'  font-family:\'Press Start 2P\',cursive;font-size:9px;'
        f'  color:#ffffff;background:{border_color};'
        f'  padding:4px 12px;letter-spacing:0.5px;'
        f'  white-space:nowrap;'
        f'">{label}</div>'
    )
    container = (
        f'<div style="'
        f'  border:2px solid {border_color};'
        f'  background:rgba(15,12,41,0.72);'
        f'  padding:10px 8px 8px 8px;'
        f'  display:flex;gap:0;align-items:center;'
        f'">{sprites_row}</div>'
    )
    return (
        f'<div style="position:relative;flex:1;">'
        f'  {tab}'
        f'  {container}'
        f'</div>'
    )

_rival_label  = "TU EQUIPO" if _challenge_mode else "EQUIPO RIVAL"
_panel_ia     = _make_team_panel(_team_ia,    _active_ia,    "#00d4ff", "EQUIPO IA")
_panel_rival  = _make_team_panel(_team_rival, _active_rival, "#ff4b4b", _rival_label)

st.html(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    </style>
    <div style="display:flex;gap:20px;padding-top:24px;padding-bottom:6px;">
      {_panel_ia}
      {_panel_rival}
    </div>
    """
)

st.divider()

col_stats, col_log, col_actions = st.columns([1, 1.2, 1])

with col_stats:
    st.subheader("📊 Comparativa")
    st.table(pd.DataFrame({"IA (Aliado)": current_ia["stats"], "Rival": current_rival["stats"]}))
    # Coloured type badges for both active Pokémon
    st.markdown(
        "**IA:** " + "".join(type_badge_html(t) for t in current_ia["types"]) +
        "<br>**Rival:** " + "".join(type_badge_html(t) for t in current_rival["types"]),
        unsafe_allow_html=True,
    )
    if current_ia.get("status"):
        st.markdown(f"IA status: {status_badge_html(current_ia['status'])}", unsafe_allow_html=True)
    if current_rival.get("status"):
        st.markdown(f"Rival status: {status_badge_html(current_rival['status'])}", unsafe_allow_html=True)

with col_log:
    st.subheader("📜 Registro de Combate")
    with st.container(height=320):
        for entry in st.session_state.historial:
            st.write(entry)

with col_actions:
    if st.session_state.battle_finished:
        st.success(st.session_state.resultado)

    # ── Post-faint forced switch (challenge mode only) ──────────────────────
    elif st.session_state.get("must_switch_rival") and mode == "2. Desafío":
        st.subheader("💀 ¡Tu Pokémon se ha debilitado!")
        st.write("Elige tu siguiente Pokémon:")
        for idx, pokemon in enumerate(_team_rival):
            if pokemon["debilitado"]:
                continue
            hp_pct = int(pokemon.get("current_hp", 1.0) * 100)
            type_html = "".join(type_badge_html(t, small=True) for t in pokemon["types"])
            btn_label = f"➡️ {pokemon['name']}  ({hp_pct}% HP)"
            if st.button(btn_label, key=f"forcedswitch_{idx}", use_container_width=True):
                _send_in_pokemon("rival", idx)
                st.session_state.must_switch_rival = False
                st.rerun()
            st.markdown(type_html, unsafe_allow_html=True)

    elif mode == "2. Desafío":
        st.subheader("🕹️ Tus Ataques")

        # ── Mega / Primal button ───────────────────────────────────────────
        # Rules:
        #   • Only one mega per side per battle (rival_mega_used tracks this)
        #   • Button disappears once confirmed — cannot be undone
        #   • Primal Reversion (Kyogre/Groudon) uses different label
        _rival_item_name = (current_rival.get("item") or {}).get("name", "")
        _rival_item_key  = _rival_item_name.lower().strip().replace(" ", "-")
        _is_primal       = _rival_item_key in ("red-orb", "blue-orb")
        _can_mega_rival  = (
            not st.session_state.get("rival_mega_used", False)
            and not current_rival.get("mega_evolved", False)
            and not current_rival.get("debilitado", False)
            and _rival_item_key in MEGA_STONE_MAP
        )
        _mega_pending = st.session_state.get("rival_mega_confirmed", False)

        if _can_mega_rival:
            if not _mega_pending:
                # Choose label depending on whether it's Primal Reversion
                _btn_label = (
                    f"🌊 EFECTO PRIMIGENIO ({_rival_item_name})"
                    if _is_primal else
                    f"⚡ MEGA EVOLUCIONAR ({_rival_item_name})"
                )
                if st.button(
                    _btn_label,
                    use_container_width=True,
                    help="Pulsa para activar y luego elige un ataque. No se puede deshacer.",
                ):
                    st.session_state["rival_mega_confirmed"] = True
                    st.rerun()
            else:
                # Confirmed — show locked confirmation, no way to undo
                _confirmed_label = "🌊 Efecto Primigenio activado" if _is_primal else "⚡ Mega Evolución activada"
                st.success(f"{_confirmed_label} — elige tu ataque")

        # ── Shared move-button chrome (injected once per render) ───────────
        # The CSS uses the :has() pseudo-class (Chrome 105+, Firefox 121+,
        # Safari 15.4+) to target the st.button() element immediately after
        # each hidden marker div, applying the per-type background colour.
        # This makes the ENTIRE button surface clickable while keeping full
        # type-colour styling — no separate action button needed.
        st.markdown("""
        <style>
        /* Shared chrome for ALL move buttons in this section */
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button {
            border: 1px solid rgba(255,255,255,0.20) !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            text-align: left !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            white-space: pre-line !important;
            line-height: 1.55 !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.28) !important;
            min-height: 58px !important;
            width: 100% !important;
            transition: filter 0.14s ease, transform 0.10s ease !important;
        }
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button:hover {
            filter: brightness(1.14) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.35) !important;
        }
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button:active {
            filter: brightness(0.92) !important;
            transform: translateY(0px) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        for idx, move in enumerate(current_rival["moves"]):
            effectiveness = get_type_multiplier(move.get("type"), current_ia.get("types", []))
            eff_label     = describe_effectiveness(effectiveness)

            type_name  = (move.get("type") or "normal").lower().strip()
            colors     = get_type_colors(type_name)
            bg, fg     = colors["bg"], colors["text"]
            emoji      = get_type_emoji(type_name)
            type_label = type_name.capitalize()
            move_name  = (move.get("name") or "???").replace("-", " ").title()
            power      = move.get("power")
            pwr_str    = f"PWR {power}" if power else "Status"

            # Effectiveness suffix — only shown for non-neutral hits
            eff_map = {
                "Super effective":   "  ✦ Super effective!",
                "Not very effective": "  ▼ Not very effective",
                "Immune":            "  ✕ No effect",
            }
            eff_suffix = eff_map.get(eff_label, "")

            # ── Inject hidden marker + per-button colour override ──────────
            # The :has(#mvmkrN) rule below overrides ONLY this button's
            # bg/fg while the shared rule above handles shape, size, hover.
            st.markdown(
                f"""<style>
div[data-testid="stMarkdownContainer"]:has(#mvmkr{idx}) \
+ div[data-testid="stButton"] button {{
    background-color: {bg} !important;
    color: {fg} !important;
}}
</style>
<div id="mvmkr{idx}" style="display:none;height:0;overflow:hidden;"></div>""",
                unsafe_allow_html=True,
            )

            # ── The entire button IS the card ──────────────────────────────
            # Line 1: type emoji + type label (left)   move name (right-ish)
            # Line 2: power or status info             effectiveness note
            btn_label = (
                f"{emoji} {type_label:<10}  {move_name}\n"
                f"    {pwr_str}{eff_suffix}"
            )

            if st.button(
                btn_label,
                key=f"at_{idx}",
                use_container_width=True,
                help=get_move_tooltip(move, current_ia),
            ):
                # ── Apply Mega Evolution before the turn (if toggle was on) ──
                if _mega_pending:
                    # _team_rival is a direct reference to the engine's internal
                    # list (get_state() returns self._team_rival, not a copy).
                    # Likewise, _team_rival[_active_rival] is the SAME dict object
                    # as the engine's self._rival_pokemon, so updating it in-place
                    # guarantees the engine uses the new mega stats during step().
                    _active_data = _team_rival[_active_rival]
                    _held = (_active_data.get("item") or {}).get("name", "")
                    _mega_data = _apply_form_transforms(_active_data, _held)
                    _mega_data["mega_evolved"] = True
                    _active_data.update(_mega_data)   # mutate in-place — engine sees it
                    st.session_state["rival_mega_used"]     = True
                    st.session_state["rival_mega_confirmed"] = False  # clear for next battle

                # ── IA switching logic ────────────────────────────────────
                # Before attacking, check if the IA should switch to a
                # better-matchup Pokémon.  Only fires when the current
                # matchup is clearly unfavourable AND a better target exists.
                _should_switch, _switch_target = get_ia_switch_decision(
                    st.session_state.env
                )
                if _should_switch and _switch_target is not None:
                    switch_ia_pokemon(_switch_target, action_rival=idx)
                else:
                    ia_action = predict_action_compatible(
                        st.session_state.loaded_model, st.session_state.env
                    )
                    combat_step(ia_action, action_rival=idx)
                st.rerun()

        st.divider()
        st.subheader("🔁 Cambiar Pokémon")
        switch_options = get_switch_options(_team_rival, _active_rival)
        if not switch_options:
            st.caption("No hay otros Pokémon disponibles para cambiar.")
        else:
            for switch_idx, switch_label in switch_options:
                if st.button(switch_label, key=f"switch_{switch_idx}", use_container_width=True):
                    switch_rival_pokemon(switch_idx)
                    st.rerun()
    else:
        if not auto:
            st.warning("⏸️ Simulación pausada.")
        elif not st.session_state.battle_finished:
            # ── IA switching logic (simulation mode) ─────────────────────
            _should_switch_sim, _switch_target_sim = get_ia_switch_decision(
                st.session_state.env
            )
            if _should_switch_sim and _switch_target_sim is not None:
                switch_ia_pokemon(_switch_target_sim)
            else:
                ia_action = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
                combat_step(ia_action)
            time.sleep(speed)
            st.rerun()


if st.session_state.battle_finished:
    st.divider()
    st.header("📊 Informe Analítico Post-Combate")
    try:
        conn = sqlite3.connect("pokemon_bigdata.db")
        df_hp = pd.read_sql("SELECT id, hp_ia, hp_rival FROM v_logs ORDER BY id ASC", conn)
        if not df_hp.empty:
            st.subheader("📈 Evolución de Vitalidad")
            st.line_chart(df_hp.set_index("id"))

            col_ia_report, col_rival_report = st.columns(2)
            with col_ia_report:
                st.subheader("⚔️ Movimientos IA")
                df_ia = pd.read_sql(
                    """
                    SELECT ia_move_name AS Movimiento, ia_move_type AS Tipo, ia_effectiveness AS Efectividad, COUNT(*) AS Usos
                    FROM v_logs
                    GROUP BY ia_move_name, ia_move_type, ia_effectiveness
                    """,
                    conn,
                )
                st.dataframe(df_ia, use_container_width=True)
            with col_rival_report:
                st.subheader("🛡️ Movimientos Rival")
                df_rival = pd.read_sql(
                    """
                    SELECT rival_move AS Movimiento, rival_move_type AS Tipo, rival_effectiveness AS Efectividad, COUNT(*) AS Usos
                    FROM v_logs
                    GROUP BY rival_move, rival_move_type, rival_effectiveness
                    """,
                    conn,
                )
                st.dataframe(df_rival, use_container_width=True)
        conn.close()
    except Exception as exc:
        st.error(f"Error cargando informe: {exc}")

    if st.button("🔄 REINICIAR TODO", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()


with st.expander("📊 Explorador de Big Data (Dataset Completo)"):
    try:
        conn = sqlite3.connect("pokemon_bigdata.db")
        df_full = pd.read_sql("SELECT * FROM pokemon_stats", conn)
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Total Pokémon Ingeridos", len(df_full))
        if not df_full.empty:
            metric_col_2.metric("Tipo más común", df_full["type1"].mode()[0].capitalize())
            st.dataframe(df_full, use_container_width=True)
        else:
            st.warning("La tabla está vacía. Ejecuta el ETL primero.")
        conn.close()
    except Exception:
        st.info("Consejo: ejecuta `etl_process.py` para cargar los datos de la API en SQL y ver esta sección.")
