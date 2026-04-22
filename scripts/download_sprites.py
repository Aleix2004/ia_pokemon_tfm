"""
download_sprites.py
~~~~~~~~~~~~~~~~~~~
One-time offline sprite downloader for the Pokémon sprite system.

Downloads sprites from the Pokémon Showdown CDN and builds a local
sprite index (assets/sprites/sprite_index.json) used by src/sprites.py
for zero-latency, fully offline sprite resolution.

Run this script ONCE during project setup.  After it completes,
src.sprites.SPRITE_INDEX_LOADED will be True and the dashboard + RL
training pipeline will serve sprites from local files — no HTTP needed.

USAGE
─────
    # Download sprites for the training roster only (~6 Pokémon, fast)
    python scripts/download_sprites.py --roster-only

    # Download ALL Pokémon from the Showdown dex (~1000 Pokémon, ~5 min)
    python scripts/download_sprites.py --full-dex

    # Download from a custom slug list
    python scripts/download_sprites.py --slugs my_slugs.txt

    # Custom output directory
    python scripts/download_sprites.py --full-dex --output-dir /data/sprites

    # More parallel threads (faster on good connections)
    python scripts/download_sprites.py --full-dex --threads 32

OUTPUT
──────
    assets/sprites/
      ani/            {slug}.gif   animated front
      ani-back/       {slug}.gif   animated back
      gen5/           {slug}.png   static front
      gen5-back/      {slug}.png   static back
      fallback/
        unknown.png               placeholder for missing sprites
      sprite_index.json           precomputed lookup table

SPRITE INDEX FORMAT
───────────────────
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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SHOWDOWN_BASE = "https://play.pokemonshowdown.com/sprites"
SHOWDOWN_DEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"

POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon"

# Sprite types to attempt for each Pokémon, in priority order.
# Normal sprites
SPRITE_TYPES_NORMAL: list[tuple[str, str, str]] = [
    # (type_key,       cdn_path_prefix,  local_extension)
    ("ani",            "ani",            ".gif"),
    ("ani-back",       "ani-back",       ".gif"),
    ("gen5",           "gen5",           ".png"),
    ("gen5-back",      "gen5-back",      ".png"),
]
# Shiny variants — same slugs, different CDN subdirectory
SPRITE_TYPES_SHINY: list[tuple[str, str, str]] = [
    ("ani-shiny",      "ani-shiny",      ".gif"),
    ("ani-back-shiny", "ani-back-shiny", ".gif"),
    ("gen5-shiny",     "gen5-shiny",     ".png"),
    ("gen5-back-shiny","gen5-back-shiny",".png"),
]
# Default: all types (overridden at runtime by --no-shiny flag)
SPRITE_TYPES: list[tuple[str, str, str]] = SPRITE_TYPES_NORMAL + SPRITE_TYPES_SHINY

# ── Known mega/gmax/primal/regional form slugs (Showdown merged format) ──────
# Derived from the MEGA_STONE_MAP and GMAX_POKEMON in src/pokemon_forms.py.
# These are the slugs Showdown uses for their sprite files.
MEGA_FORM_SLUGS: list[str] = [
    # Mega Evolutions
    "venusaurmega", "charizardmegax", "charizardmegay", "blastoisemega",
    "alakazammega", "gengarmega", "kangaskhanmega", "pinsirmega",
    "gyaradosmega", "aerodactylmega", "mewtwomegax", "mewtwomegay",
    "amphароsmega", "steelixmega", "scizormega", "heracrossmega",
    "houndoommega", "tyranitarmega", "slowbromega", "sharpedomega",
    "cameruptmega", "altariamega", "banettемega", "absolmega",
    "glaliemega", "salamencemega", "metagrossmega", "latiasmega",
    "latiosmega", "rayquazamega", "lopunnymega", "garchompmega",
    "lucariomega", "abomasnowmega", "gallademega", "audinomega",
    "diancimega", "sceptilemega", "blazikenmega", "swampertmega",
    "gardevoirmega", "sableyemega", "mawilemega", "aggronmega",
    "medichamega", "manectricmega", "sharpеdomega", "swampertmega",
    "pidgeotmega", "beedrillmega",
    # Primal Reversions
    "kyogreprimal", "groudonprimal",
    # Gigantamax
    "charizardgmax", "venusaurgmax", "blastoisegmax", "butterfreeGmax",
    "pikachugmax", "meowthgmax", "machampgmax", "gengargmax",
    "kinglergmax", "laprasgmax", "eeveeGmax", "snorlaxgmax",
    "corviknightgmax", "orbeetlegmax", "drednawgmax", "coalossalgmax",
    "flapeggmax", "sandacondagmax", "centiskorchgmax", "alcremiegmax",
    "grimmsnarlegmax", "hatterenegmax", "toxtricitygmax",
    "copperajahgmax", "duraludongmax", "eternatuseternamax",
    "urshifugmax", "urshifurapidstrikegmax",
    # Alolan forms
    "raichualola", "sandshrewalola", "sandslashalola", "vulpixalola",
    "ninetalesalola", "diglettalola", "dugtrioalola", "meowthalola",
    "persianalola", "geodudealola", "graveleralola", "golemalola",
    "grimeralola", "mukalola", "exeggutoralola", "marowakalola",
    # Galarian forms
    "meowthgalar", "ponytagalar", "rapidashgalar", "farfetchdgalar",
    "weezinggalar", "mrmimegalar", "corsolагalar", "zigzagoongalar",
    "linoonegalar", "darumakaGalar", "darmanitangalar",
    "yamilaskgalar", "stunfiskgalar",
    # Hisuian forms
    "growlithehisui", "arcaninehisui", "voltorbhisui", "electrodehisui",
    "typhlosionhisui", "qwilfishhisui", "sneaselhisui", "samurotthisui",
    "lilliganthisui", "zoroarkhisui", "braviaryhisui", "sliggoohisui",
    "goodrahisui", "avalugghisui", "decidueyehisui",
]

# Project root = two levels up from this script (scripts/ → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPRITE_DIR = _PROJECT_ROOT / "assets" / "sprites"
_INDEX_FILE = "sprite_index.json"

# Timeout per HTTP request (seconds)
_TIMEOUT = 8

# Known-good fallback image: Showdown's own silhouette (#0)
_FALLBACK_URL = f"{SHOWDOWN_BASE}/gen5/0.png"
_FALLBACK_LOCAL = "fallback/unknown.png"


# ─────────────────────────────────────────────────────────────────────────────
#  SLUG SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def get_roster_slugs() -> list[str]:
    """
    Return Showdown slugs for the TRAINING_ROSTER defined in pokemon_env.py.
    Falls back to a hardcoded minimal set if the module can't be imported.
    """
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.env.pokemon_env import TRAINING_ROSTER
        from src.pokemon_forms import normalize_showdown_name
        return [normalize_showdown_name(p["name"]) for p in TRAINING_ROSTER]
    except Exception:
        # Hardcoded fallback — matches the default TRAINING_ROSTER
        return [
            "charizard", "blastoise", "venusaur",
            "pikachu", "garchomp", "alakazam",
        ]


def get_showdown_dex_slugs(session: requests.Session) -> list[str]:
    """
    Fetch all Pokémon slugs from Showdown's own dex JSON.

    This is a one-time network call during setup only.  Returns slugs
    exactly as Showdown uses them (already lowercase, hyphenated).
    """
    print("  Fetching Showdown dex for full Pokémon list...", end=" ", flush=True)
    try:
        r = session.get(SHOWDOWN_DEX_URL, timeout=15)
        r.raise_for_status()
        dex = r.json()
        slugs = sorted(dex.keys())
        print(f"got {len(slugs)} entries")
        return slugs
    except Exception as exc:
        print(f"FAILED ({exc})")
        print("  Falling back to roster-only slugs")
        return get_roster_slugs()


def load_slugs_from_file(path: str) -> list[str]:
    """Load one slug per line from a text file, stripping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _cdn_url(sprite_type: str, slug: str, ext: str) -> str:
    return f"{SHOWDOWN_BASE}/{sprite_type}/{slug}{ext}"


def download_sprite(
    session:     requests.Session,
    slug:        str,
    sprite_dir:  Path,
) -> list[str]:
    """
    Download all available sprite types for *slug*.

    Returns a list of type_keys that were successfully downloaded
    (e.g. ["ani", "ani-back", "gen5", "gen5-back"] or a subset).
    """
    available: list[str] = []

    for type_key, cdn_prefix, ext in SPRITE_TYPES:
        url      = _cdn_url(cdn_prefix, slug, ext)
        dest_dir = sprite_dir / cdn_prefix
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest     = dest_dir / f"{slug}{ext}"

        # Skip if already downloaded (resume-friendly)
        if dest.exists() and dest.stat().st_size > 0:
            available.append(type_key)
            continue

        try:
            r = session.get(url, timeout=_TIMEOUT, stream=True)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                available.append(type_key)
        except Exception:
            pass  # silent — this type simply won't appear in the index

    return available


# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def ensure_fallback_sprite(session: requests.Session, sprite_dir: Path) -> None:
    """
    Download (or create) the fallback placeholder image.

    Tries to download Showdown's own silhouette (#0).  If that fails,
    creates a minimal 1×1 transparent PNG so the file always exists.
    """
    fallback_dir = sprite_dir / "fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    dest = fallback_dir / "unknown.png"

    if dest.exists() and dest.stat().st_size > 0:
        return

    print("  Downloading fallback placeholder...", end=" ", flush=True)
    try:
        r = session.get(_FALLBACK_URL, timeout=_TIMEOUT)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            print("OK")
            return
    except Exception:
        pass

    # Minimal 1×1 transparent PNG (67 bytes, no external dependency)
    _TINY_PNG = bytes([
        0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,  # PNG signature
        0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,  # IHDR chunk length + type
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,  # width=1, height=1
        0x08,0x06,0x00,0x00,0x00,0x1F,0x15,0xC4,  # 8-bit RGBA, CRC
        0x89,0x00,0x00,0x00,0x0A,0x49,0x44,0x41,  # IDAT chunk
        0x54,0x78,0x9C,0x62,0x00,0x00,0x00,0x02,
        0x00,0x01,0xE2,0x21,0xBC,0x33,0x00,0x00,
        0x00,0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,  # IEND chunk
        0x60,0x82,
    ])
    dest.write_bytes(_TINY_PNG)
    print("created minimal placeholder")


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_index_from_disk(sprite_dir: Path) -> dict[str, list[str]]:
    """
    Scan ALL sprite directories (normal + shiny) and build the index from
    what's actually on disk.

    This is more reliable than trusting download results — it catches
    partial downloads, resumed runs and manually added sprites.
    """
    index: dict[str, list[str]] = {}

    # Scan every known type dir — both normal and shiny
    all_types = SPRITE_TYPES_NORMAL + SPRITE_TYPES_SHINY
    for type_key, cdn_prefix, ext in all_types:
        type_dir = sprite_dir / cdn_prefix
        if not type_dir.exists():
            continue
        for file in type_dir.iterdir():
            if file.suffix.lower() == ext and file.stat().st_size > 0:
                slug = file.stem   # filename without extension
                if slug not in index:
                    index[slug] = []
                index[slug].append(type_key)

    return index


def write_index(sprite_dir: Path, index: dict[str, list[str]]) -> Path:
    """Write the sprite index JSON to disk.  Returns the path written."""
    payload = {
        "version":      1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count":        len(index),
        "sprites":      index,
    }
    out = sprite_dir / _INDEX_FILE
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PROGRESS REPORTER
# ─────────────────────────────────────────────────────────────────────────────

class _Progress:
    """Minimal progress reporter (uses tqdm if available, else plain print)."""

    def __init__(self, total: int, desc: str = ""):
        self._total   = total
        self._done    = 0
        self._start   = time.monotonic()
        self._tqdm    = None
        self._desc    = desc
        try:
            from tqdm import tqdm as _tqdm
            self._tqdm = _tqdm(total=total, desc=desc, unit="sprite", ncols=80)
        except ImportError:
            print(f"\n  {desc} — {total} Pokémon (install tqdm for progress bar)")

    def update(self, slug: str, n_types: int) -> None:
        self._done += 1
        if self._tqdm:
            self._tqdm.set_postfix({"last": slug, "types": n_types})
            self._tqdm.update(1)
        elif self._done % 50 == 0 or self._done == self._total:
            elapsed = time.monotonic() - self._start
            pct = 100 * self._done / max(1, self._total)
            print(f"  {self._done}/{self._total} ({pct:.0f}%)  {elapsed:.0f}s", flush=True)

    def close(self) -> None:
        if self._tqdm:
            self._tqdm.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DOWNLOAD ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def download_all(
    slugs:      list[str],
    sprite_dir: Path,
    threads:    int = 16,
) -> dict[str, list[str]]:
    """
    Download sprites for all slugs concurrently.

    Parameters
    ----------
    slugs       : Showdown slug strings (e.g. ["mr-mime", "charizard"])
    sprite_dir  : root directory for downloaded sprites
    threads     : number of parallel download threads

    Returns
    -------
    Index dict built from what's actually on disk after download.
    """
    sprite_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "PokemonRLTraining/1.0 sprite-downloader"
    # Keep-alive pool for N threads
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=threads,
        pool_maxsize=threads * 2,
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)

    ensure_fallback_sprite(session, sprite_dir)

    # Deduplicate preserving order
    seen:        set[str] = set()
    unique_slugs: list[str] = []
    for s in slugs:
        if s and s not in seen:
            seen.add(s)
            unique_slugs.append(s)

    print(f"\n  Downloading sprites for {len(unique_slugs)} Pokémon "
          f"({threads} threads) ...\n")

    progress = _Progress(len(unique_slugs), desc="Downloading")

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(download_sprite, session, slug, sprite_dir): slug
            for slug in unique_slugs
        }
        for future in as_completed(futures):
            slug  = futures[future]
            types = future.result()
            if types:
                results[slug] = types
            progress.update(slug, len(types))

    progress.close()

    # Rebuild from disk (catches any previously downloaded files)
    on_disk = build_index_from_disk(sprite_dir)
    # Merge: on_disk takes precedence (more reliable)
    for slug, types in results.items():
        if slug not in on_disk:
            on_disk[slug] = types

    return on_disk


# ─────────────────────────────────────────────────────────────────────────────
#  POKEAPI MEGA DOWNLOADER
#
#  Showdown CDN returns 404 for ALL mega/primal/gmax form slugs.
#  PokeAPI stores official artwork for every form, accessible by PokeAPI
#  form name (e.g. "charizard-mega-x") at:
#    https://pokeapi.co/api/v2/pokemon/{form-name}
#    → response["sprites"]["front_default"]  (GitHub raw URL)
#    → response["sprites"]["back_default"]   (GitHub raw URL)
#
#  Files are saved using the Showdown-merged slug format so the sprite
#  registry can find them with the same normalize_showdown_name() function.
#    "charizard-mega-x" → "charizardmegax.png"
# ─────────────────────────────────────────────────────────────────────────────

def _merge_slug(name: str) -> str:
    """
    Convert a PokeAPI form name to the merged Showdown slug used by
    sprite_registry.py (all non-alphanumeric chars removed).

    Examples
      "charizard-mega-x"  → "charizardmegax"
      "venusaur-mega"     → "venusaurmega"
      "kyogre-primal"     → "kyogreprimal"
      "charizard-gmax"    → "charizardgmax"
    """
    import re as _re
    s = name.strip().lower()
    s = _re.sub(r"[^a-z0-9]", "", s)
    return s or "unknown"


def _get_pokeapi_form_names() -> list[tuple[str, str]]:
    """
    Return (pokeapi_form_name, merged_slug) pairs for all mega/primal/gmax forms.
    Derived from MEGA_STONE_MAP and GMAX_POKEMON in src/pokemon_forms.py.
    """
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.pokemon_forms import MEGA_STONE_MAP, GMAX_POKEMON
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        # Mega / Primal forms
        for spec in MEGA_STONE_MAP.values():
            form_name = spec["form_name"]   # e.g. "charizard-mega-x"
            slug      = _merge_slug(form_name)
            if slug not in seen:
                seen.add(slug)
                pairs.append((form_name, slug))
        # G-Max forms
        for form_name in GMAX_POKEMON.values():
            slug = _merge_slug(form_name)
            if slug not in seen:
                seen.add(slug)
                pairs.append((form_name, slug))
        return pairs
    except Exception as exc:
        print(f"  ⚠  Could not import pokemon_forms: {exc}")
        # Hardcoded minimal fallback
        return [
            ("charizard-mega-x",  "charizardmegax"),
            ("charizard-mega-y",  "charizardmegay"),
            ("venusaur-mega",     "venusaurmega"),
            ("blastoise-mega",    "blastoisemega"),
            ("mewtwo-mega-x",     "mewtwomegax"),
            ("mewtwo-mega-y",     "mewtwomegay"),
        ]


def _fetch_pokeapi_sprite_urls(
    session:    requests.Session,
    form_name:  str,
) -> dict[str, str | None]:
    """
    Call PokeAPI for *form_name* and return front/back sprite URLs.

    Returns dict with keys "front" and "back"; values are URLs or None.
    """
    url = f"{POKEAPI_BASE}/{form_name}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 404:
            return {"front": None, "back": None, "error": "404"}
        r.raise_for_status()
        sprites = r.json().get("sprites", {})
        return {
            "front": sprites.get("front_default"),
            "back":  sprites.get("back_default"),
        }
    except Exception as exc:
        return {"front": None, "back": None, "error": str(exc)}


def download_mega_from_pokeapi(
    sprite_dir: Path,
    threads:    int = 8,
) -> dict[str, list[str]]:
    """
    Download mega/primal/gmax sprites from PokeAPI and save them locally
    using Showdown-merged slug filenames.

    Returns an index dict: {slug: [type_keys_downloaded]}
    """
    pairs = _get_pokeapi_form_names()
    print(f"\n  Fetching {len(pairs)} mega/primal/gmax forms from PokeAPI...")
    print("  (This may take a minute — one API call per form)")

    session = requests.Session()
    session.headers["User-Agent"] = "PokemonRLTraining/1.0 sprite-downloader"

    # Step 1: collect all sprite URLs from PokeAPI (serial, rate-limit friendly)
    url_map: list[tuple[str, str, str | None, str | None]] = []
    ok = 0
    missing = 0
    for i, (form_name, slug) in enumerate(pairs, 1):
        result = _fetch_pokeapi_sprite_urls(session, form_name)
        front  = result.get("front")
        back   = result.get("back")
        if front or back:
            url_map.append((form_name, slug, front, back))
            ok += 1
        else:
            err = result.get("error", "no sprites")
            print(f"  ✗ {form_name:30s} → {err}")
            missing += 1
        # Gentle rate-limiting: 2 requests/sec to avoid hammering PokeAPI
        if i % 10 == 0:
            print(f"    {i}/{len(pairs)} API calls done...")
            time.sleep(0.5)

    print(f"\n  API calls done: {ok} with sprites, {missing} missing")

    if not url_map:
        print("  ❌  No sprites found via PokeAPI.")
        return {}

    # Step 2: download the actual image files concurrently
    gen5_dir      = sprite_dir / "gen5"
    gen5_back_dir = sprite_dir / "gen5-back"
    gen5_dir.mkdir(parents=True, exist_ok=True)
    gen5_back_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, list[str]] = {}
    lock_ok = 0
    lock_fail = 0

    def _dl_image(img_url: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 0:
            return True   # already downloaded
        try:
            r = session.get(img_url, timeout=_TIMEOUT, stream=True)
            if r.status_code == 200 and len(r.content) > 0:
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
        return False

    print(f"\n  Downloading {len(url_map)} sprite sets...")
    progress = _Progress(len(url_map), desc="PokeAPI megas")

    def _worker(args: tuple) -> tuple[str, list[str]]:
        form_name, slug, front_url, back_url = args
        got: list[str] = []
        if front_url:
            dest = gen5_dir / f"{slug}.png"
            if _dl_image(front_url, dest):
                got.append("gen5")
        if back_url:
            dest = gen5_back_dir / f"{slug}.png"
            if _dl_image(back_url, dest):
                got.append("gen5-back")
        return slug, got

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in url_map}
        for future in as_completed(futures):
            form_name = futures[future]
            slug, got = future.result()
            if got:
                downloaded[slug] = got
            progress.update(slug, len(got))

    progress.close()

    ok_count   = sum(1 for v in downloaded.values() if v)
    fail_count = len(url_map) - ok_count
    print(f"\n  Downloaded: {ok_count} Pokémon, {fail_count} failed")

    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Pokémon Showdown sprites and build sprite_index.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Slug source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--full-dex",  action="store_true",
        help="Download ALL Pokémon from Showdown's pokedex.json (~1000 Pokémon)"
    )
    src.add_argument(
        "--roster-only", action="store_true",
        help="Download only the TRAINING_ROSTER Pokémon (~6 Pokémon, fast)"
    )
    src.add_argument(
        "--slugs", metavar="FILE",
        help="Text file with one Showdown slug per line"
    )
    src.add_argument(
        "--megas-only", action="store_true",
        help="Download only Mega/GMax/regional form sprites via Showdown CDN (fast, ~100 slugs)"
    )
    src.add_argument(
        "--megas-pokeapi", action="store_true",
        help="Download Mega/Primal/GMax sprites via PokeAPI (correct source — Showdown CDN 404s for these)"
    )

    p.add_argument(
        "--include-megas", action="store_true",
        help="Also download Mega/GMax/regional forms on top of the main slug list"
    )
    p.add_argument(
        "--no-shiny", action="store_true",
        help="Skip shiny sprite variants (downloads normal sprites only)"
    )
    p.add_argument(
        "--output-dir", default=str(_DEFAULT_SPRITE_DIR),
        help="Root directory for downloaded sprites and index"
    )
    p.add_argument(
        "--threads", type=int, default=16,
        help="Number of parallel download threads"
    )
    p.add_argument(
        "--index-only", action="store_true",
        help="Skip downloading — rebuild index from files already on disk"
    )
    p.add_argument(
        "--validate", action="store_true",
        help="After building, validate every indexed file exists on disk"
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    sprite_dir = Path(args.output_dir).resolve()

    print(f"\n{'═'*60}")
    print("  Pokémon Showdown sprite downloader")
    print(f"  Output: {sprite_dir}")
    print(f"{'═'*60}")

    # ── PokeAPI mega downloader mode ──────────────────────────────────────────
    if args.megas_pokeapi:
        print("\n  Mode: PokeAPI mega/primal/gmax downloader")
        print("  (Showdown CDN returns 404 for all mega sprites — using PokeAPI instead)")
        mega_index = download_mega_from_pokeapi(sprite_dir, threads=min(args.threads, 8))
        if mega_index:
            # Merge with any existing index on disk
            full_index = build_index_from_disk(sprite_dir)
            idx_path   = write_index(sprite_dir, full_index)
            print(f"\n  ✅  {len(mega_index)} mega forms downloaded")
            print(f"      Index updated: {len(full_index)} total slugs → {idx_path}")
        else:
            print("\n  ❌  No sprites downloaded.")
        return

    # ── Index-only mode ────────────────────────────────────────────────────────
    if args.index_only:
        print("\n  Index-only mode: scanning existing files...")
        index = build_index_from_disk(sprite_dir)
        idx_path = write_index(sprite_dir, index)
        print(f"\n  ✅  Index rebuilt: {len(index)} slugs → {idx_path}")
        return

    # ── Configure which sprite types to download ───────────────────────────────
    global SPRITE_TYPES
    if args.no_shiny:
        SPRITE_TYPES = SPRITE_TYPES_NORMAL
        print("\n  Mode: normal sprites only (shiny skipped)")
    else:
        SPRITE_TYPES = SPRITE_TYPES_NORMAL + SPRITE_TYPES_SHINY
        print("\n  Mode: normal + shiny sprites")

    # ── Determine slug list ────────────────────────────────────────────────────
    if args.megas_only:
        slugs = MEGA_FORM_SLUGS
        print(f"\n  Slug source: mega/gmax/regional forms ({len(slugs)} slugs)")
    elif args.slugs:
        slugs = load_slugs_from_file(args.slugs)
        print(f"\n  Slug source: {args.slugs} ({len(slugs)} slugs)")
    elif args.roster_only:
        slugs = get_roster_slugs()
        print(f"\n  Slug source: TRAINING_ROSTER ({len(slugs)} slugs)")
    else:
        # Default: full dex
        session_tmp = requests.Session()
        slugs = get_showdown_dex_slugs(session_tmp)
        session_tmp.close()
        print(f"\n  Slug source: Showdown full dex ({len(slugs)} slugs)")

    # Optionally append mega slugs on top of any source
    if args.include_megas and not args.megas_only:
        extra = [s for s in MEGA_FORM_SLUGS if s not in set(slugs)]
        slugs = list(slugs) + extra
        print(f"  + {len(extra)} mega/gmax/regional form slugs appended")

    if not slugs:
        print("  ❌  No slugs to download.  Exiting.")
        sys.exit(1)

    # ── Download ───────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    index = download_all(slugs, sprite_dir, threads=args.threads)
    elapsed = time.monotonic() - t0

    # ── Write index ────────────────────────────────────────────────────────────
    idx_path = write_index(sprite_dir, index)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_files = sum(len(v) for v in index.values())
    print(f"\n{'─'*60}")
    print(f"  Download complete in {elapsed:.1f}s")
    print(f"  Pokémon with sprites : {len(index)}")
    print(f"  Total files on disk  : {total_files}")
    print(f"  Index written to     : {idx_path}")
    print(f"{'─'*60}")

    # ── Type breakdown ─────────────────────────────────────────────────────────
    for type_key, _, _ in SPRITE_TYPES:
        n = sum(1 for v in index.values() if type_key in v)
        print(f"  {type_key:<12} : {n} sprites")

    # ── Optional validation ────────────────────────────────────────────────────
    if args.validate:
        print("\n  Validating: checking every indexed file exists on disk...")
        bad = 0
        for slug, types in index.items():
            for type_key, cdn_prefix, ext in SPRITE_TYPES:
                if type_key in types:
                    p = sprite_dir / cdn_prefix / f"{slug}{ext}"
                    if not p.exists():
                        print(f"  MISSING: {p}")
                        bad += 1
        if bad:
            print(f"\n  ❌  {bad} missing files detected")
        else:
            print(f"\n  ✅  All files present on disk")

    print(f"\n  Next step: the sprite system is ready.")
    print(f"  src.sprites.SPRITE_INDEX_LOADED will be True.")
    print(f"  The dashboard will now serve sprites locally with zero latency.\n")


if __name__ == "__main__":
    main()
