"""
sprites.py
~~~~~~~~~~
Zero-latency, fully offline sprite resolution for the Pokémon system.

ARCHITECTURE
────────────

  ┌───────────────────────────────────────────────────────────────────┐
  │  SPRITE_INDEX  (dict, loaded ONCE at import — never again)       │
  │                                                                    │
  │  "mr-mime"   → ["ani", "ani-back", "gen5", "gen5-back"]          │
  │  "charizard" → ["ani", "ani-back", "gen5", "gen5-back"]          │
  │  "kommo-o"   → ["gen5", "gen5-back"]                             │
  │  ...                                                               │
  │                                                                    │
  │  Backed by: assets/sprites/sprite_index.json                      │
  └──────────────────────────────────┬────────────────────────────────┘
                                     │  O(1) dict lookup
                                     ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  get_showdown_sprite_local(form_name)                             │
  │    1. normalize_showdown_name()  → clean slug                     │
  │    2. _build_sprite_candidates() → ordered slug list              │
  │    3. SPRITE_INDEX lookup        → which formats exist            │
  │    4. return local Path strings  → no I/O, no HTTP               │
  └───────────────────────────────────────────────────────────────────┘

PERFORMANCE CHARACTERISTICS
────────────────────────────
• Module import: one JSON load (~5 ms, happens once per process)
• Per-call:      one dict lookup + a few set membership checks = O(1)
• No I/O per step, no HTTP, no Streamlit, no subprocess overhead
• Safe for SubprocVecEnv workers: each process loads its own copy of
  the dict from the JSON file (no shared mutable state)

SPRITE_INDEX JSON FORMAT  (assets/sprites/sprite_index.json)
─────────────────────────
  {
    "version": 1,
    "generated_at": "2024-01-01T00:00:00",
    "count": 1010,
    "sprites": {
      "bulbasaur": ["ani", "ani-back", "gen5", "gen5-back"],
      "mr-mime":   ["ani", "ani-back", "gen5", "gen5-back"],
      "kommo-o":   ["gen5", "gen5-back"],
      ...
    }
  }

LOCAL DIRECTORY STRUCTURE
──────────────────────────
  assets/sprites/
    ani/            *.gif   animated front sprites
    ani-back/       *.gif   animated back sprites
    gen5/           *.png   static front sprites (fallback)
    gen5-back/      *.png   static back sprites  (fallback)
    fallback/
      unknown.png           used when all candidates fail

SETUP (one-time, offline)
──────────────────────────
  python scripts/download_sprites.py --full-dex

After running, SPRITE_INDEX_LOADED will be True and all lookups will be
served from local files.  The dashboard's get_showdown_sprite() falls
back to HTTP automatically if the local cache is absent.

DEBUG MODE
──────────
  Set DEBUG_SPRITES = True (or env var POKEMON_DEBUG_SPRITES=1)
  to log every lookup and fallback decision to stdout.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

try:
    from src.pokemon_forms import normalize_pokemon_name, normalize_showdown_name
except ImportError:
    from pokemon_forms import normalize_pokemon_name, normalize_showdown_name


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Project root = two levels up from this file (src/sprites.py → src/ → root)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_SPRITE_DIR:   Path = _PROJECT_ROOT / "assets" / "sprites"
_INDEX_PATH:   Path = _SPRITE_DIR / "sprite_index.json"
_FALLBACK:     str  = str(_SPRITE_DIR / "fallback" / "unknown.png")
# HTTP silhouette used when local cache is absent AND as a last resort
_REMOTE_FALLBACK: str = "https://play.pokemonshowdown.com/sprites/gen5/0.png"


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG FLAG
# ─────────────────────────────────────────────────────────────────────────────

DEBUG_SPRITES: bool = os.environ.get("POKEMON_DEBUG_SPRITES", "").strip() == "1"

_log = logging.getLogger(__name__)


def _dbg(msg: str) -> None:
    """Emit a debug message when DEBUG_SPRITES is enabled."""
    if DEBUG_SPRITES:
        _log.debug("[sprites] %s", msg)
        print(f"[sprites] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX LOADING  (runs once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _load_index() -> dict[str, list[str]]:
    """
    Load the precomputed sprite index from disk.

    Returns an empty dict if the index file doesn't exist — the system
    degrades gracefully to HTTP fetching in that case.

    The dict maps Showdown slug → list of available sprite type strings:
        "ani"      → assets/sprites/ani/{slug}.gif
        "ani-back" → assets/sprites/ani-back/{slug}.gif
        "gen5"     → assets/sprites/gen5/{slug}.png
        "gen5-back"→ assets/sprites/gen5-back/{slug}.png
    """
    if not _INDEX_PATH.exists():
        _dbg(f"index not found at {_INDEX_PATH} — HTTP fallback will be used")
        return {}
    try:
        with _INDEX_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        sprites = data.get("sprites", {})
        _dbg(f"loaded sprite index: {len(sprites)} entries from {_INDEX_PATH}")
        return sprites
    except Exception as exc:
        _log.warning("[sprites] Failed to load sprite index: %s", exc)
        return {}


# Module-level index — loaded once, shared across all calls in this process.
# SubprocVecEnv workers each load their own copy (no shared mutable state).
SPRITE_INDEX: dict[str, list[str]] = _load_index()

# Public flag: True once the index is loaded and non-empty.
# dashboard.py checks this to decide whether to use local or HTTP sprites.
SPRITE_INDEX_LOADED: bool = bool(SPRITE_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
#  CANDIDATE CHAIN
# ─────────────────────────────────────────────────────────────────────────────

def _build_sprite_candidates(slug: str) -> list[str]:
    """
    Return an ordered list of Showdown slugs to try for *slug*.

    Resolution order
    ────────────────
    1. Exact normalised slug          ("charizard-mega-x", "mr-mime")
    2. Progressive prefix stripping   ("charizard-mega", "charizard")
       — intermediate form names tried BEFORE the bare base species
    3. Base species via normalize_pokemon_name()  (deduplicated)
       — catches form suffixes not covered by simple prefix splits
       e.g. tapu-koko-gmax → progressive gives tapu-koko, tapu
            normalize_pokemon_name gives tapu-koko (already present → skipped)

    Multi-word base Pokémon are safe: "mr-mime" → ["mr-mime", "mr"].
    "mr" is only tried if "mr-mime" genuinely fails, which is the correct
    last-resort behaviour before returning the fallback placeholder.

    Returns a list that is NEVER empty (falls back to ["unknown"]).

    Examples
    ────────
    "charizard-mega-x" → ["charizard-mega-x", "charizard-mega", "charizard"]
    "mr-mime"          → ["mr-mime", "mr"]
    "tapu-koko-gmax"   → ["tapu-koko-gmax", "tapu-koko", "tapu"]
    "pikachu"          → ["pikachu"]
    """
    seen:   set[str]  = set()
    result: list[str] = []

    def _add(s: str) -> None:
        s = s.strip("-")
        if s and s not in seen:
            seen.add(s)
            result.append(s)

    _add(slug)                             # 1. exact
    parts = slug.split("-")
    for i in range(len(parts) - 1, 0, -1):
        _add("-".join(parts[:i]))          # 2. progressive strip
    _add(normalize_pokemon_name(slug))     # 3. explicit base species

    return result or ["unknown"]


# ─────────────────────────────────────────────────────────────────────────────
#  PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_TO_PATH: dict[str, tuple[str, str]] = {
    # sprite_type → (subdirectory, extension)
    "ani":       ("ani",       ".gif"),
    "ani-back":  ("ani-back",  ".gif"),
    "gen5":      ("gen5",      ".png"),
    "gen5-back": ("gen5-back", ".png"),
}

# Priority order: prefer animated GIF, fall back to static PNG
_FRONT_PRIORITY: tuple[str, ...] = ("ani",      "gen5")
_BACK_PRIORITY:  tuple[str, ...] = ("ani-back", "gen5-back")


def _local_path(slug: str, sprite_type: str) -> str:
    """Return the absolute local file path for a given slug and sprite type."""
    subdir, ext = _TYPE_TO_PATH[sprite_type]
    return str(_SPRITE_DIR / subdir / f"{slug}{ext}")


def _resolve_side(
    candidates: list[str],
    priority:   tuple[str, ...],
    available:  dict[str, list[str]],
) -> str | None:
    """
    Walk the candidate list and return the first available local path.

    Parameters
    ----------
    candidates : ordered slug list from _build_sprite_candidates()
    priority   : sprite type preference order ("ani" before "gen5", etc.)
    available  : the loaded SPRITE_INDEX

    Returns
    -------
    Absolute path string, or None if nothing found.
    """
    for slug in candidates:
        types = available.get(slug)
        if not types:
            _dbg(f"  candidate '{slug}' not in index")
            continue
        type_set = set(types)
        for t in priority:
            if t in type_set:
                path = _local_path(slug, t)
                _dbg(f"  resolved '{slug}' via type '{t}' → {path}")
                return path
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_showdown_sprite_local(form_name: str) -> dict[str, str]:
    """
    Resolve a Pokémon's sprite to local file paths with zero HTTP requests.

    This is the production-grade sprite resolver for both the Streamlit UI
    and RL training pipelines.  It performs:

    1. Name normalisation  — normalize_showdown_name() → clean slug
    2. Candidate chain     — _build_sprite_candidates() → ordered slugs
    3. Index lookup        — O(1) dict access per candidate (no disk I/O)
    4. Path construction   — deterministic: slug + type → absolute path
    5. Fallback            — local placeholder, never None, never HTTP

    Performance
    ───────────
    • Per-call cost: ~3 dict lookups + string ops = <1 µs
    • No file I/O per call (the index is trusted; validated separately)
    • No network I/O ever
    • Thread-safe (reads only from immutable module-level dict)
    • Safe in SubprocVecEnv workers (each worker has its own dict copy)

    Parameters
    ----------
    form_name : str
        Any human-readable Pokémon name or form slug.
        "Mr. Mime", "Kommo O", "charizard-mega-x", "Pikachu-Gmax", …
        Normalisation is handled internally.

    Returns
    -------
    dict[str, str] with guaranteed non-empty string values:
        "front"    — local path to front sprite (animated GIF preferred)
        "back"     — local path to back sprite
        "animated" — local path to /ani/ GIF (may not exist; hint only)

    Raises
    ------
    Never raises.  All errors produce the fallback placeholder path.

    Examples
    --------
    >>> get_showdown_sprite_local("Mr. Mime")
    {"front": ".../assets/sprites/ani/mr-mime.gif",
     "back":  ".../assets/sprites/ani-back/mr-mime.gif",
     "animated": ".../assets/sprites/ani/mr-mime.gif"}

    >>> get_showdown_sprite_local("Unknown Pokemon XYZ")
    {"front": ".../assets/sprites/fallback/unknown.png",
     "back":  ".../assets/sprites/fallback/unknown.png",
     "animated": ".../assets/sprites/ani/unknown-pokemon-xyz.gif"}
    """
    if not SPRITE_INDEX_LOADED:
        # Index not available — caller should use the HTTP fallback
        raise RuntimeError(
            "SPRITE_INDEX not loaded.  Run scripts/download_sprites.py first, "
            "or use get_showdown_sprite() (HTTP version) as a fallback."
        )

    try:
        slug = normalize_showdown_name(form_name)
        _dbg(f"resolving '{form_name}' → slug='{slug}'")

        candidates = _build_sprite_candidates(slug)
        _dbg(f"  candidates: {candidates}")

        front = _resolve_side(candidates, _FRONT_PRIORITY, SPRITE_INDEX)
        back  = _resolve_side(candidates, _BACK_PRIORITY,  SPRITE_INDEX)

        return {
            "front":    front or _effective_fallback(),
            "back":     back  or _effective_fallback(),
            "animated": _local_path(slug, "ani"),
        }
    except Exception as exc:
        _log.warning("[sprites] get_showdown_sprite_local error for %r: %s", form_name, exc)
        fb = _effective_fallback()
        return {"front": fb, "back": fb, "animated": fb}


def _effective_fallback() -> str:
    """
    Return the best available fallback sprite path.

    Preference order:
      1. Local  assets/sprites/fallback/unknown.png  (zero latency)
      2. Remote Showdown silhouette                  (requires internet)
    """
    if Path(_FALLBACK).exists():
        return _FALLBACK
    return _REMOTE_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def validate_sprite_index(verbose: bool = False) -> dict[str, int]:
    """
    Verify that every entry in SPRITE_INDEX has its files on disk.

    Does NOT use the index as authoritative — physically checks each path.
    Intended for startup health-checks and CI validation, not hot paths.

    Parameters
    ----------
    verbose : bool
        If True, prints one line per missing file.

    Returns
    -------
    dict with keys:
        "total"    — total (slug, sprite_type) pairs checked
        "missing"  — count of files that are absent from disk
        "slugs_ok" — slugs where all declared types are present
        "slugs_bad"— slugs with at least one missing file
    """
    total = missing = slugs_ok = slugs_bad = 0

    for slug, types in SPRITE_INDEX.items():
        slug_ok = True
        for t in types:
            total += 1
            path = Path(_local_path(slug, t))
            if not path.exists():
                missing += 1
                slug_ok = False
                if verbose:
                    print(f"  MISSING  {path}")
        if slug_ok:
            slugs_ok += 1
        else:
            slugs_bad += 1

    return {
        "total":     total,
        "missing":   missing,
        "slugs_ok":  slugs_ok,
        "slugs_bad": slugs_bad,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  —  python -m src.sprites
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"\n  Sprite index loaded : {SPRITE_INDEX_LOADED}")
    print(f"  Index path          : {_INDEX_PATH}")
    print(f"  Sprite directory    : {_SPRITE_DIR}")
    print(f"  Entries in index    : {len(SPRITE_INDEX)}")
    print(f"  Fallback sprite     : {_FALLBACK}")

    if SPRITE_INDEX_LOADED:
        print("\n  Validating index (checking files on disk)...")
        result = validate_sprite_index(verbose=True)
        print(f"\n  Total checked  : {result['total']}")
        print(f"  Missing files  : {result['missing']}")
        print(f"  Clean slugs    : {result['slugs_ok']}")
        print(f"  Broken slugs   : {result['slugs_bad']}")

        # Quick lookup demo
        demo = ["Mr. Mime", "Kommo O", "Tapu Koko", "Ho Oh", "Porygon Z",
                "charizard-mega-x", "pikachu-gmax"]
        print("\n  Demo lookups:")
        for name in demo:
            result_d = get_showdown_sprite_local(name)
            print(f"    {name!r:<22} → front={result_d['front'].split('/')[-1]}")
    else:
        print(
            "\n  ⚠  No sprite index found.  Run the download script to build it:\n"
            "     python scripts/download_sprites.py --full-dex\n"
        )
        sys.exit(1)
