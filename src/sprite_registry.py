"""
sprite_registry.py
~~~~~~~~~~~~~~~~~~
Production-grade, RL-safe, deterministic sprite asset registry.

══════════════════════════════════════════════════════════════════════════════
IMPORT RULE — ENFORCED BY DESIGN
══════════════════════════════════════════════════════════════════════════════

    Importing this module has ZERO side effects.

    No JSON is parsed. No files are opened. No globals are populated.
    The module is safe to import in any context — test runners, RL workers,
    Streamlit, CLI scripts — without triggering any I/O.

    Initialization is ALWAYS explicit:

        from src.sprite_registry import init_sprite_registry
        meta = init_sprite_registry()          # call this once, early

    After that, all lookups are O(1) and do zero I/O.

══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  IMPORT TIME  — zero side effects                                    │
  │                                                                      │
  │  _STATE = _RegistryState()    ← empty, uninitialized                │
  │  _OVERRIDES = {...}           ← pure constant dict, no I/O          │
  │  normalize_showdown_name()    ← pure function, no state             │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │  EXPLICIT CALL (once, before workers)
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  init_sprite_registry(path, expected_version)                        │
  │    1. Read sprite_index.json                                         │
  │    2. Compute sha256   → RegistryMeta (version + hash)              │
  │    3. Build dict[slug → SpritePaths]                                 │
  │    4. Wrap in MappingProxyType  → immutable forever                 │
  │    5. Set _STATE.initialized = True                                  │
  │    6. Return RegistryMeta  (store in experiment config!)            │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │  Linux / macOS (fork)               │  Windows (spawn)
          │                                     │
          │  SubprocVecEnv inherits             │  Workers start fresh.
          │  _STATE via fork().                 │  PokemonEnv.__init__
          │  Copy-on-write: _STATE never        │  calls init_sprite_registry()
          │  mutated → zero pages copied.       │  (idempotent, ~10 ms once).
          │  True zero duplication.             │  Never during step loop.
          └──────────────────┬──────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────────┐
  │  get_sprite(form_name) → SpritePaths                                 │
  │    normalize_showdown_name()  → slug            (pure, no state)    │
  │    _STATE.registry.get(slug)  → SpritePaths     (O(1) dict lookup)  │
  │    _FALLBACK singleton        → if slug not in registry (data miss)  │
  │                                                                      │
  │  NEVER: disk I/O, network I/O, url construction, HTTP requests      │
  └──────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
MULTIPROCESSING STRATEGY
══════════════════════════════════════════════════════════════════════════════

  CHOSEN STRATEGY: Pre-spawn initialization + idempotent worker re-init.

  This is the only strategy that is:
    ✔ correct under both fork and spawn
    ✔ zero I/O during the training step loop
    ✔ no shared mutable state (MappingProxyType is immutable)
    ✔ no IPC overhead (no Manager, no Queue, no mmap)

  USAGE PATTERN (train_ppo.py):

      # 1. Initialize in main process BEFORE SubprocVecEnv creation
      from src.sprite_registry import init_sprite_registry
      meta = init_sprite_registry(args.sprite_index)

      # 2. Log meta for reproducibility
      exp_config["sprite_registry"] = meta._asdict()

      # 3. Create vectorized envs — workers inherit state (fork) or
      #    re-initialize in their own __init__ (spawn)
      envs = SubprocVecEnv([make_env(i, cfg) for i in range(n_envs)])

  USAGE PATTERN (PokemonEnv.__init__):

      from src.sprite_registry import init_sprite_registry, is_initialized
      if not is_initialized():
          # Only runs in spawn workers — fork workers already have _STATE
          init_sprite_registry(self.cfg.sprite_index_path)

  WHY NOT shared_memory / mmap:
    The sprite index is ~1 MB of string data.  Shared memory gives
    ~30% memory reduction but adds 200+ lines of platform-specific code
    and a persistent OS resource that must be cleaned up.  For a 1 MB
    read-only dict, per-process copies under spawn are acceptable.
    Under fork (Linux/macOS), COW eliminates the duplication entirely.

══════════════════════════════════════════════════════════════════════════════
REPRODUCIBILITY SYSTEM
══════════════════════════════════════════════════════════════════════════════

  init_sprite_registry() returns RegistryMeta with:
    • version      — semantic version string from index JSON
    • generated_at — ISO timestamp of index generation
    • count        — number of registered slugs
    • sha256       — SHA-256 of the index file content (hex, 64 chars)
    • path         — absolute resolved path to the index file

  Store RegistryMeta._asdict() in your experiment YAML/JSON config.
  To verify a replay: call init_sprite_registry(expected_version="X")
  and compare sha256 values.  Mismatch → fail fast.

══════════════════════════════════════════════════════════════════════════════
SETUP (one-time, offline)
══════════════════════════════════════════════════════════════════════════════

  python scripts/download_sprites.py --full-dex
  → assets/sprites/{ani,ani-back,gen5,gen5-back}/*.{gif,png}
  → assets/sprites/sprite_index.json

  After setup, init_sprite_registry() will find the index.
  The fallback image (assets/sprites/fallback/unknown.png) must exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  (constants — computed at import, no I/O, no side effects)
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_SPRITE_DIR:   Path = _PROJECT_ROOT / "assets" / "sprites"

# Public constant — default location of the versioned sprite index.
DEFAULT_INDEX_PATH: str = str(_SPRITE_DIR / "sprite_index.json")

# Local fallback — must exist after running download_sprites.py.
# This path is baked into _FALLBACK at import time; no I/O per call.
FALLBACK_PATH: str = str(_SPRITE_DIR / "fallback" / "unknown.png")


# ─────────────────────────────────────────────────────────────────────────────
#  VALUE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class SpritePaths(NamedTuple):
    """
    Immutable container for a Pokémon's resolved local sprite paths.

    NamedTuple chosen over dataclass because:
      • Immutable by construction — mutations raise AttributeError
      • Zero per-call allocation when returned from the registry dict
      • Picklable for any future multiprocessing use
      • dict-compatible via ._asdict()

    All fields are absolute path strings pointing to local files.
    """
    front: str   # Animated GIF preferred; falls back to static PNG
    back:  str   # Animated GIF preferred; falls back to static PNG
    ani:   str   # /ani/ GIF path (hint; front is preferred for rendering)


class RegistryMeta(NamedTuple):
    """
    Reproducibility snapshot returned by init_sprite_registry().

    Store ._asdict() in your experiment YAML/JSON for full reproducibility.
    Compare sha256 values to verify identical asset sets across runs.

    Fields
    ------
    version      : semantic version string from sprite_index.json
    generated_at : ISO 8601 timestamp of when the index was built
    count        : number of Pokémon slugs registered
    sha256       : SHA-256 hex digest of the sprite_index.json file content
    path         : absolute resolved path to the sprite_index.json file
    """
    version:      str
    generated_at: str
    count:        int
    sha256:       str
    path:         str


# Module-level fallback singleton — allocated ONCE at import, reused forever.
# Never construct a new SpritePaths in the hot path.
_FALLBACK: SpritePaths = SpritePaths(
    front=FALLBACK_PATH,
    back=FALLBACK_PATH,
    ani=FALLBACK_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
#  REGISTRY STATE  (empty at import — NO side effects)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RegistryState:
    """
    Internal mutable singleton.  All fields are None / False at import time.

    This is the ONLY mutable global in this module.  It is written to exactly
    once per process: during init_sprite_registry().  After that it is
    read-only in all callers — including concurrent Streamlit threads and
    forked RL worker processes.

    Under fork: child processes inherit the populated state via copy-on-write.
    Under spawn: child processes start with an empty (uninitialized) state and
    call init_sprite_registry() from their own PokemonEnv.__init__.

    Fields
    ------
    registry       : Normal sprite lookup (slug → SpritePaths).
    shiny_registry : Shiny sprite lookup (slug → SpritePaths).
                     May be empty if no shiny sprites are downloaded.
                     Never None after initialization.
    meta           : Reproducibility snapshot (sha256, version, …).
    initialized    : Set to True exactly once by init_sprite_registry().
    """
    registry:       MappingProxyType | None = None
    shiny_registry: MappingProxyType | None = None
    meta:           RegistryMeta | None     = None
    initialized:    bool                    = False


# The singleton.  Empty at import.  Zero I/O.  No side effects.
_STATE: _RegistryState = _RegistryState()

# Threading lock — protects concurrent calls from Streamlit threads.
# This is a threading.Lock (not multiprocessing.Lock) — it only guards within
# a single process.  Each spawned worker process has its own lock instance.
# IMPORTANT: init_sprite_registry() must be called BEFORE creating
# SubprocVecEnv, so no forked child ever holds this lock at fork time.
_INIT_LOCK: threading.Lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALIZATION — normalize_showdown_name()
# ─────────────────────────────────────────────────────────────────────────────
#
# DESIGN INVARIANTS
# ─────────────────
# • Pure function: no global state read or written, deterministic.
# • All Pokémon-specific exceptions are in _OVERRIDES (a constant dict).
#   No ad-hoc per-Pokémon code outside _OVERRIDES.
# • Regex objects compiled at import time — zero compile overhead per call.
# • Identical output on every platform (Windows/Linux/macOS), every process,
#   every seed.  This function is used identically by RL env and Streamlit UI.

_OVERRIDES: dict[str, str] = {
    # ══════════════════════════════════════════════════════════════════════════
    # CANONICAL SLUG FORMAT
    # ══════════════════════════════════════════════════════════════════════════
    # Showdown sprite *files* use merged lowercase slugs with NO separators:
    #   mrmime.png   nidoranf.png   hooh.png   irontreads.png   tapukoko.png
    #
    # PokeAPI uses hyphenated slugs:   mr-mime  nidoran-f  ho-oh
    # Display names may use spaces:   "Mr. Mime"  "Tapu Koko"  "Iron Treads"
    #
    # Every entry here maps BOTH the PokeAPI form AND the display form to the
    # merged slug that matches the actual sprite filename on disk.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Gender symbols ────────────────────────────────────────────────────────
    "nidoran♀":            "nidoranf",
    "nidoran♂":            "nidoranm",
    "nidoran-f":           "nidoranf",
    "nidoran-m":           "nidoranm",
    "nidoran f":           "nidoranf",
    "nidoran m":           "nidoranm",

    # ── Mr. Mime family ───────────────────────────────────────────────────────
    "mr. mime":            "mrmime",
    "mr mime":             "mrmime",
    "mr-mime":             "mrmime",
    "mr. mime-galar":      "mrmimegalar",
    "mr mime galar":       "mrmimegalar",
    "mr-mime-galar":       "mrmimegalar",
    "mime jr.":            "mimejr",
    "mime jr":             "mimejr",
    "mime-jr":             "mimejr",

    # ── Type: Null ────────────────────────────────────────────────────────────
    "type: null":          "typenull",
    "type:null":           "typenull",
    "type-null":           "typenull",
    "type null":           "typenull",

    # ── Apostrophes (straight and curly) ─────────────────────────────────────
    "farfetch'd":          "farfetchd",
    "farfetch\u2019d":     "farfetchd",
    "farfetchd":           "farfetchd",
    "sirfetch'd":          "sirfetchd",
    "sirfetch\u2019d":     "sirfetchd",
    "sirfetchd":           "sirfetchd",

    # ── Accented characters ───────────────────────────────────────────────────
    "flabébé":             "flabebe",
    "flabebe":             "flabebe",

    # ── Ho-Oh / Porygon-Z (hyphens are part of the name, not form separators) ─
    "ho-oh":               "hooh",
    "ho oh":               "hooh",
    "porygon-z":           "porygonz",
    "porygon z":           "porygonz",
    "porygon2":            "porygon2",

    # ── Pseudo-legendary Dragon families with -o suffix ───────────────────────
    "jangmo-o":            "jangmoo",
    "jangmo o":            "jangmoo",
    "hakamo-o":            "hakamoo",
    "hakamo o":            "hakamoo",
    "kommo-o":             "kommoo",
    "kommo o":             "kommoo",

    # ── Tapu quartet ─────────────────────────────────────────────────────────
    "tapu koko":           "tapukoko",
    "tapu-koko":           "tapukoko",
    "tapu lele":           "tapulele",
    "tapu-lele":           "tapulele",
    "tapu bulu":           "tapubulu",
    "tapu-bulu":           "tapubulu",
    "tapu fini":           "tapufini",
    "tapu-fini":           "tapufini",

    # ── Ruinous quartet (Gen IX) ──────────────────────────────────────────────
    "wo-chien":            "wochien",
    "wo chien":            "wochien",
    "chien-pao":           "chienpao",
    "chien pao":           "chienpao",
    "ting-lu":             "tinglu",
    "ting lu":             "tinglu",
    "chi-yu":              "chiyu",
    "chi yu":              "chiyu",

    # ── Paradox Pokémon — Past ────────────────────────────────────────────────
    "great tusk":          "greattusk",
    "great-tusk":          "greattusk",
    "scream tail":         "screamtail",
    "scream-tail":         "screamtail",
    "brute bonnet":        "brutebonnet",
    "brute-bonnet":        "brutebonnet",
    "flutter mane":        "fluttermane",
    "flutter-mane":        "fluttermane",
    "slither wing":        "slitherwing",
    "slither-wing":        "slitherwing",
    "sandy shocks":        "sandyshocks",
    "sandy-shocks":        "sandyshocks",
    "roaring moon":        "roaringmoon",
    "roaring-moon":        "roaringmoon",
    "walking wake":        "walkingwake",
    "walking-wake":        "walkingwake",
    "gouging fire":        "gougingfire",
    "gouging-fire":        "gougingfire",
    "raging bolt":         "ragingbolt",
    "raging-bolt":         "ragingbolt",

    # ── Paradox Pokémon — Future ──────────────────────────────────────────────
    "iron treads":         "irontreads",
    "iron-treads":         "irontreads",
    "iron bundle":         "ironbundle",
    "iron-bundle":         "ironbundle",
    "iron hands":          "ironhands",
    "iron-hands":          "ironhands",
    "iron jugulis":        "ironjugulis",
    "iron-jugulis":        "ironjugulis",
    "iron moth":           "ironmoth",
    "iron-moth":           "ironmoth",
    "iron thorns":         "ironthorns",
    "iron-thorns":         "ironthorns",
    "iron valiant":        "ironvaliant",
    "iron-valiant":        "ironvaliant",
    "iron leaves":         "ironleaves",
    "iron-leaves":         "ironleaves",
    "iron boulder":        "ironboulder",
    "iron-boulder":        "ironboulder",
    "iron crown":          "ironcrown",
    "iron-crown":          "ironcrown",

    # ── Paldea legendaries / misc ─────────────────────────────────────────────
    "terapagos":           "terapagos",
    "pecharunt":           "pecharunt",
    "ogerpon":             "ogerpon",
    "ursaluna":            "ursaluna",
    "ursaluna-bloodmoon":  "ursalunabloodmoon",
    "ursaluna bloodmoon":  "ursalunabloodmoon",
}

# Compiled at import time — zero regex compilation cost per normalize() call.
_RE_WHITESPACE:   re.Pattern[str] = re.compile(r"\s+")
_RE_NON_SLUG:     re.Pattern[str] = re.compile(r"[^a-z0-9\-]")
_RE_MULTI_HYPHEN: re.Pattern[str] = re.compile(r"-{2,}")


def normalize_showdown_name(name: str) -> str:
    """
    Convert any Pokémon name to its canonical Pokémon Showdown sprite slug.

    SINGLE normalization function used identically by:
      • RL environment (PokemonEnv)
      • Streamlit UI (dashboard.py, via get_sprite())
      • sprite registry (this module)
      • download script (scripts/download_sprites.py)

    Output is guaranteed identical on every platform, process, and seed.

    Pipeline
    ────────
    1.  strip() + lower()           canonical input form
    2.  _OVERRIDES lookup           deterministic edge-case table
    3.  Remove . ' ' :              punctuation absent from Showdown slugs
    4.  \\s+ → ''                   spaces removed (Showdown merges multi-word names)
    5.  [^a-z0-9] → ''             strip all remaining non-alphanumeric chars
    6.  empty → 'unknown'           guaranteed non-empty return

    KEY RULE — Showdown merges ALL compound names without separators:
      • Base Pokémon multi-word names  : Mr. Mime    → mrmime
      • Form variants                  : charizard-mega-x → charizardmegax
      • Regional forms                 : raichu-alola     → raichualola
      • Paradox Pokémon                : Iron Treads      → irontreads
    All known tricky cases (gender symbols, accents, apostrophes) are caught
    by the _OVERRIDES table before reaching the generic pipeline.

    Examples
    ────────
    "Mr. Mime"          → "mrmime"
    "Ho Oh"             → "hooh"
    "Nidoran♀"          → "nidoranf"
    "Type: Null"        → "typenull"
    "Farfetch'd"        → "farfetchd"
    "charizard-Mega-X"  → "charizardmegax"
    "Venusaur-Mega"     → "venusaurmega"
    "Tapu Koko"         → "tapukoko"
    "Iron Treads"       → "irontreads"
    "raichu-alola"      → "raichualola"
    "CHARIZARD"         → "charizard"
    ""                  → "unknown"
    """
    key = name.strip().lower()

    # Step 2 — override table (gender symbols, accents, apostrophes, etc.)
    if key in _OVERRIDES:
        return _OVERRIDES[key]

    s = key
    # Step 3 — remove punctuation characters
    s = s.replace(".", "").replace("'", "").replace("\u2019", "").replace(":", "")
    # Step 4 — remove spaces (Showdown merges words: "tapu koko" → "tapukoko")
    s = _RE_WHITESPACE.sub("", s)
    # Step 5 — remove hyphens and any remaining non-alphanumeric chars
    #   Showdown merges all compound forms: charizard-mega-x → charizardmegax
    s = re.sub(r"[^a-z0-9]", "", s)
    # Step 6 — empty guard
    return s or "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  PATH CONSTRUCTION  (pure functions, no state)
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_META: dict[str, tuple[str, str]] = {
    # Normal sprites
    "ani":            ("ani",            ".gif"),
    "ani-back":       ("ani-back",       ".gif"),
    "gen5":           ("gen5",           ".png"),
    "gen5-back":      ("gen5-back",      ".png"),
    # Shiny sprites — stored in parallel subdirs with identical filenames
    "ani-shiny":      ("ani-shiny",      ".gif"),
    "ani-back-shiny": ("ani-back-shiny", ".gif"),
    "gen5-shiny":     ("gen5-shiny",     ".png"),
    "gen5-back-shiny":("gen5-back-shiny",".png"),
}

_FRONT_PRIORITY:       tuple[str, ...] = ("ani",            "gen5")
_BACK_PRIORITY:        tuple[str, ...] = ("ani-back",       "gen5-back")
_SHINY_FRONT_PRIORITY: tuple[str, ...] = ("ani-shiny",      "gen5-shiny")
_SHINY_BACK_PRIORITY:  tuple[str, ...] = ("ani-back-shiny", "gen5-back-shiny")


def _local_path(slug: str, sprite_type: str) -> str:
    subdir, ext = _TYPE_META[sprite_type]
    return str(_SPRITE_DIR / subdir / f"{slug}{ext}")


# ─────────────────────────────────────────────────────────────────────────────
#  REGISTRY CONSTRUCTION  (called ONCE inside init_sprite_registry)
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry(
    raw: dict[str, list[str]],
) -> tuple[dict[str, SpritePaths], dict[str, SpritePaths]]:
    """
    Pre-compute SpritePaths for every slug in the raw index.

    Returns a 2-tuple: (normal_registry, shiny_registry).

    Single-pass construction — no runtime logic deferred.
    ──────────────────────────────────────────────────────
    For every slug present in the downloaded sprite index, resolve its best
    available sprite type (animated GIF preferred, static PNG as fallback)
    and store the result as a pre-built SpritePaths namedtuple.

    After this function returns:
    • Both registries are plain dict[str, SpritePaths].
    • Every value was constructed here — no path building ever happens
      during a get_sprite() call.
    • Both dicts are immediately wrapped in MappingProxyType by
      init_sprite_registry() and are never mutated again.

    SHINY REGISTRY
    ──────────────
    Slugs with at least one shiny type key (ani-shiny, gen5-shiny, …) are
    added to shiny_registry with paths pointing to the shiny subdirs.
    Slugs without shiny sprites are absent from shiny_registry; get_sprite()
    falls through to the normal registry for those.

    DESIGN CONTRACT — game-engine asset model
    ──────────────────────────────────────────
    This registry is the complete sprite lookup table.  Every slug that the
    RL environment or UI may request MUST appear here.  Missing slugs return
    _FALLBACK — which is a DATA error (incomplete index), not a runtime error.

    The download script (scripts/download_sprites.py --full-dex) is
    responsible for building a comprehensive index that covers all Pokémon
    forms used during training.  If a new form is added to the game, the
    index must be regenerated and the registry re-initialized with the new
    version — the sha256 in RegistryMeta tracks this.
    """
    registry:       dict[str, SpritePaths] = {}
    shiny_registry: dict[str, SpritePaths] = {}

    for slug, types in raw.items():
        type_set = set(types)

        # ── Normal paths ─────────────────────────────────────────────────────
        front = next(
            (_local_path(slug, t) for t in _FRONT_PRIORITY if t in type_set),
            FALLBACK_PATH,
        )
        back = next(
            (_local_path(slug, t) for t in _BACK_PRIORITY if t in type_set),
            FALLBACK_PATH,
        )
        ani = _local_path(slug, "ani")   # /ani/ hint — front used for rendering
        registry[slug] = SpritePaths(front=front, back=back, ani=ani)

        # ── Shiny paths (only when at least one shiny type exists) ────────────
        has_shiny = any(
            t in type_set for t in (*_SHINY_FRONT_PRIORITY, *_SHINY_BACK_PRIORITY)
        )
        if has_shiny:
            s_front = next(
                (_local_path(slug, t) for t in _SHINY_FRONT_PRIORITY if t in type_set),
                front,   # fall back to normal front if no shiny front
            )
            s_back = next(
                (_local_path(slug, t) for t in _SHINY_BACK_PRIORITY if t in type_set),
                back,    # fall back to normal back if no shiny back
            )
            s_ani = _local_path(slug, "ani-shiny")
            shiny_registry[slug] = SpritePaths(front=s_front, back=s_back, ani=s_ani)

    return registry, shiny_registry


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC INITIALIZATION API
# ─────────────────────────────────────────────────────────────────────────────

def init_sprite_registry(
    path: str = DEFAULT_INDEX_PATH,
    *,
    expected_version: str | None = None,
) -> RegistryMeta:
    """
    Initialize the sprite registry from a versioned, hashed sprite index.

    This is the ONLY entry point for registry initialization.  It must be
    called explicitly before get_sprite() can be used.  Importing this module
    does NOT call it.

    WHEN TO CALL  (strict single-entrypoint rule)
    ─────────────
    • EXACTLY ONCE in main() — before any SubprocVecEnv, before any env.
    • Streamlit: once via @st.cache_resource at app startup.
    • Test fixtures: once in conftest.py / setUp().

    NEVER call from:
    • PokemonEnv.__init__ — env must be a pure simulation, zero init
    • Callbacks / wrappers — registry must already exist when they run
    • Worker processes directly — use load_registry_snapshot() instead

    Spawn workers receive a pre-built registry via load_registry_snapshot(),
    which accepts bytes produced by export_registry_snapshot() — no JSON
    re-parsing, no file I/O, no JSON decode cost per worker.

    THREAD SAFETY
    ─────────────
    Thread-safe via threading.Lock.  Concurrent calls from Streamlit threads
    are serialized; only the first call does real work.

    PROCESS SAFETY
    ──────────────
    Under fork: _STATE is inherited. _STATE.initialized = True → early return.
    Under spawn: _STATE is empty. The call does one JSON parse per worker.
    Never call this from inside the RL step loop.

    IDEMPOTENCY
    ───────────
    Calling with the same path and version a second time is a no-op that
    returns the existing RegistryMeta immediately (no re-parse, no I/O).

    VERSION ENFORCEMENT
    ────────────────────
    If expected_version is provided, raises ValueError if the index version
    doesn't match.  Use this to pin training runs to a specific asset version.

    REPRODUCIBILITY
    ───────────────
    The returned RegistryMeta includes a sha256 of the index file content.
    Store meta._asdict() in your experiment config YAML/JSON.  To verify a
    replay run, compare sha256 values.

    Parameters
    ----------
    path : str
        Path to sprite_index.json.  Defaults to DEFAULT_INDEX_PATH.
        Passed through Path.resolve() — relative paths work.
    expected_version : str | None
        If provided, init raises ValueError when the index version string
        does not equal this value.  Pass your training config's
        sprite_registry.version here.

    Returns
    -------
    RegistryMeta
        Immutable snapshot: version, generated_at, count, sha256, path.
        Never None.  Always valid when this function returns normally.

    Raises
    ------
    FileNotFoundError
        sprite_index.json not found at the resolved path.
    ValueError
        expected_version provided but doesn't match the index version.
    RuntimeError
        Index file is invalid JSON, or missing required fields.
    """
    with _INIT_LOCK:
        # ── Idempotent: already initialized ──────────────────────────────────
        if _STATE.initialized:
            if expected_version is not None:
                if _STATE.meta.version != expected_version:
                    raise ValueError(
                        f"[sprite_registry] Version mismatch: registry was "
                        f"initialized with version={_STATE.meta.version!r}, "
                        f"but expected_version={expected_version!r}. "
                        f"Ensure all processes use the same sprite_index.json."
                    )
            return _STATE.meta

        # ── Resolve and read the index file ──────────────────────────────────
        index_path = Path(path).resolve()
        if not index_path.exists():
            raise FileNotFoundError(
                f"[sprite_registry] Sprite index not found: {index_path}\n"
                f"Run: python scripts/download_sprites.py --full-dex"
            )

        raw_bytes = index_path.read_bytes()
        sha256    = hashlib.sha256(raw_bytes).hexdigest()

        # ── Parse JSON ───────────────────────────────────────────────────────
        try:
            data = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[sprite_registry] Invalid JSON in sprite index: {exc}"
            ) from exc

        version      = str(data.get("version", "unknown"))
        generated_at = str(data.get("generated_at", "unknown"))
        raw_sprites  = data.get("sprites", {})

        if not isinstance(raw_sprites, dict):
            raise RuntimeError(
                f"[sprite_registry] 'sprites' field must be a dict, "
                f"got {type(raw_sprites).__name__}"
            )

        # ── Version gate ─────────────────────────────────────────────────────
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"[sprite_registry] Version mismatch in {index_path}: "
                f"file version={version!r}, expected={expected_version!r}"
            )

        # ── Build immutable registries ────────────────────────────────────────
        registry, shiny_registry = _build_registry(raw_sprites)

        meta = RegistryMeta(
            version=version,
            generated_at=generated_at,
            count=len(registry),
            sha256=sha256,
            path=str(index_path),
        )

        # MappingProxyType prevents all mutations after this point.
        # Assignment order: registries first, then meta, then initialized.
        # This is safe because _INIT_LOCK serializes all access.
        _STATE.registry       = MappingProxyType(registry)
        _STATE.shiny_registry = MappingProxyType(shiny_registry)
        _STATE.meta           = meta
        _STATE.initialized    = True

        _log.info(
            "[sprite_registry] Ready: %d slugs | version=%s | sha256=%s…",
            meta.count, meta.version, meta.sha256[:16],
        )
        return meta


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC QUERY API
# ─────────────────────────────────────────────────────────────────────────────

def get_sprite(form_name: str, *, shiny: bool = False) -> SpritePaths:
    """
    Resolve any Pokémon name (including form/mega/gmax slugs) to local sprite paths.

    PRIORITY CHAIN
    ──────────────
    When shiny=True the lookup walks the following chain (stopping at first hit):
      1.  shiny sprite for the exact form slug  (e.g. "charizard-mega-x" in shiny)
      2.  shiny sprite for the base species     (e.g. "charizard" in shiny)
      3.  normal sprite for the exact form slug
      4.  normal sprite for the base species
      5.  _FALLBACK singleton

    When shiny=False (default):
      1.  normal sprite for the exact form slug
      2.  normal sprite for the base species
      3.  _FALLBACK singleton

    Base-species stripping removes known form suffixes using the explicit
    INVALID_FORM_SUFFIXES table from pokemon_forms — never splits on the first
    hyphen (which would corrupt "mr-mime", "ho-oh", "tapu-koko", etc.).

    GUARANTEES
    ──────────
    • Never raises, never returns None.
    • Zero disk I/O, zero network I/O, zero system calls.
    • Identical output for identical input across all processes and seeds.
    • Thread-safe and process-safe (reads from immutable MappingProxyType).
    • Graceful degradation: returns _FALLBACK if registry not yet initialized.

    Parameters
    ----------
    form_name : str
        Any Pokémon name or form slug.  Normalized internally via
        normalize_showdown_name().
    shiny : bool, optional
        When True, prefer shiny sprite variants.  Falls back gracefully
        if shiny sprites are not downloaded.  Default: False.

    Returns
    -------
    SpritePaths namedtuple with .front, .back, .ani absolute path strings.
    """
    if not _STATE.initialized:
        _log.warning(
            "[sprite_registry] get_sprite(%r, shiny=%r) called before "
            "init_sprite_registry() — returning FALLBACK.",
            form_name, shiny,
        )
        return _FALLBACK

    slug = normalize_showdown_name(form_name)

    # ── Base-species slug (strips known form suffixes) ────────────────────────
    # normalize_pokemon_name() works on the HYPHENATED PokeAPI format
    # (e.g. "charizard-mega-x" → "charizard"), but our slug is already merged
    # ("charizardmegax").  We therefore apply it to the ORIGINAL form_name and
    # then re-normalize the result, keeping the merged Showdown format.
    #
    # Example:
    #   form_name = "charizard-mega-x"
    #   slug      = "charizardmegax"          (merged, no hyphens)
    #   base_raw  = normalize_pokemon_name("charizard-mega-x") = "charizard"
    #   base_slug = normalize_showdown_name("charizard")       = "charizard"
    try:
        from src.pokemon_forms import normalize_pokemon_name as _strip_form
    except ImportError:
        from pokemon_forms import normalize_pokemon_name as _strip_form
    base_raw  = _strip_form(str(form_name))          # strips suffix on original input
    base_slug = normalize_showdown_name(base_raw)    # merge to Showdown format

    normal_reg = _STATE.registry
    shiny_reg  = _STATE.shiny_registry

    if shiny:
        # Priority 1 — shiny exact form
        result = shiny_reg.get(slug)
        if result is not None:
            return result
        # Priority 2 — shiny base species
        if base_slug != slug:
            result = shiny_reg.get(base_slug)
            if result is not None:
                return result

    # Priority 3 — normal exact form
    result = normal_reg.get(slug)
    if result is not None:
        return result

    # Priority 4 — normal base species
    if base_slug != slug:
        result = normal_reg.get(base_slug)
        if result is not None:
            return result

    # Priority 5 — _FALLBACK singleton
    return _FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
#  SPAWN-WORKER INJECTION API
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem: under Windows (spawn), each SubprocVecEnv worker starts a fresh
# Python process — _STATE.initialized = False.  Workers must NOT call
# init_sprite_registry() (that would re-parse JSON per worker, violating the
# single-entrypoint rule).
#
# Solution: main() calls export_registry_snapshot() ONCE after init, producing
# a picklable bytes object.  Each env-factory closure captures this snapshot.
# When the factory runs in the worker process, it calls load_registry_snapshot(),
# which does a fast pickle.loads() and wraps in MappingProxyType — no file I/O,
# no JSON parsing, ~1 ms per worker at startup, never during the step loop.
#
# Under fork (Linux/macOS): load_registry_snapshot() hits the
# `if _STATE.initialized: return` guard immediately (inherited state).
# Cost: one bool check per env __init__. Zero actual work.
#
# This pattern eliminates per-worker JSON parsing on all platforms while
# preserving the strict single-entrypoint initialization contract.

def export_registry_snapshot() -> bytes:
    """
    Serialize the initialized registry to a picklable bytes object.

    Call ONCE in main() after init_sprite_registry(), before creating
    SubprocVecEnv.  Pass the result to each env-factory closure so workers
    can reconstruct the registry via load_registry_snapshot() without parsing
    the JSON index file again.

    Uses pickle.HIGHEST_PROTOCOL for maximum performance.
    The snapshot contains the plain dict (not MappingProxyType, which is not
    picklable) and the RegistryMeta namedtuple.

    Returns
    -------
    bytes
        Pickled payload, typically 1–3 MB for a full Pokémon dex.

    Raises
    ------
    RuntimeError
        If init_sprite_registry() has not been called yet.
    """
    if not _STATE.initialized:
        raise RuntimeError(
            "[sprite_registry] Cannot export: registry not initialized."
        )
    import pickle
    return pickle.dumps(
        {
            "registry":       dict(_STATE.registry),        # plain dict — picklable
            "shiny_registry": dict(_STATE.shiny_registry),  # plain dict — picklable
            "meta":           _STATE.meta,                  # RegistryMeta NamedTuple
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def load_registry_snapshot(snapshot: bytes) -> None:
    """
    Reconstruct the registry in a worker process from a pre-serialized snapshot.

    Called from each env-factory thunk immediately before PokemonEnv() is
    instantiated.  Does nothing if the registry is already initialized (fork
    workers inherit the state and hit this early-return immediately).

    Under spawn:
        pickle.loads() reconstructs the dict — no JSON parsing, no file I/O.
        MappingProxyType wraps it — registry is immutable from this point.
        Cost: ~1 ms at worker startup. Zero cost during the step loop.

    Under fork:
        _STATE.initialized is True (inherited). Returns in one bool check.
        Cost: negligible.

    Parameters
    ----------
    snapshot : bytes
        Produced by export_registry_snapshot() in the main process.

    Raises
    ------
    RuntimeError
        If snapshot is malformed or incompatible.
    """
    with _INIT_LOCK:
        if _STATE.initialized:
            return   # fork: already initialized — nothing to do
        import pickle
        try:
            data = pickle.loads(snapshot)
        except Exception as exc:
            raise RuntimeError(
                f"[sprite_registry] Failed to load registry snapshot: {exc}"
            ) from exc
        _STATE.registry       = MappingProxyType(data["registry"])
        _STATE.shiny_registry = MappingProxyType(data.get("shiny_registry", {}))
        _STATE.meta           = data["meta"]
        _STATE.initialized    = True
        _log.info(
            "[sprite_registry] Loaded from snapshot: %d slugs | %d shiny | version=%s",
            len(_STATE.registry), len(_STATE.shiny_registry), _STATE.meta.version,
        )


def is_initialized() -> bool:
    """
    Return True if init_sprite_registry() or load_registry_snapshot()
    has completed successfully in this process.

    Use in test assertions and health-checks.
    Do NOT use in PokemonEnv.__init__ — envs must never initialize anything.
    """
    return _STATE.initialized


def get_registry_meta() -> RegistryMeta | None:
    """
    Return the RegistryMeta from the current initialization, or None.

    Useful for logging experiment metadata after the fact.
    """
    return _STATE.meta


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION UTILITY  (CI / health-checks only, never in hot path)
# ─────────────────────────────────────────────────────────────────────────────

def validate(verbose: bool = False) -> dict[str, int]:
    """
    Verify every SpritePaths entry in the registry points to an existing file.

    Performs real disk I/O — use only for CI/CD checks and startup
    health validation, NEVER in the training loop or per-step code.

    Returns
    -------
    dict with keys: total (pairs checked), missing (absent files),
    ok (clean slugs), bad (slugs with at least one missing file).

    Raises
    ------
    RuntimeError
        If init_sprite_registry() has not been called.
    """
    if not _STATE.initialized:
        raise RuntimeError(
            "[sprite_registry] Cannot validate: registry not initialized."
        )

    total = missing = ok = bad = 0

    for slug, paths in _STATE.registry.items():
        slug_ok = True
        for field_name in ("front", "back"):
            p = Path(getattr(paths, field_name))
            total += 1
            if not p.exists():
                missing += 1
                slug_ok = False
                if verbose:
                    print(f"  MISSING  {p}  (slug={slug!r}, field={field_name!r})")
        if slug_ok:
            ok += 1
        else:
            bad += 1

    return {"total": total, "missing": missing, "ok": ok, "bad": bad}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  —  python -m src.sprite_registry  [path]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    index_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INDEX_PATH

    try:
        meta = init_sprite_registry(index_path)
    except (FileNotFoundError, RuntimeError, ValueError) as err:
        print(f"\n  ERROR: {err}\n")
        sys.exit(1)

    print(f"\n  Registry initialized")
    print(f"  ├─ version      : {meta.version}")
    print(f"  ├─ generated_at : {meta.generated_at}")
    print(f"  ├─ count        : {meta.count} slugs")
    print(f"  ├─ sha256       : {meta.sha256}")
    print(f"  └─ path         : {meta.path}")
    print(f"\n  Fallback path   : {FALLBACK_PATH}")
    print(f"  Fallback exists : {Path(FALLBACK_PATH).exists()}")

    print("\n  Validating (physical disk check)...")
    result = validate(verbose=True)
    print(f"\n  Pairs checked   : {result['total']}")
    print(f"  Missing files   : {result['missing']}")
    print(f"  Clean slugs     : {result['ok']}")
    print(f"  Broken slugs    : {result['bad']}")

    demo_names = [
        "Mr. Mime", "Kommo O", "Ho Oh", "Nidoran♀", "Nidoran♂",
        "Type: Null", "Farfetch'd", "Tapu Koko", "Iron Treads",
        "charizard-mega-x", "pikachu-gmax", "Unknown Pokemon XYZ",
    ]
    print("\n  Demo lookups:")
    for n in demo_names:
        r = get_sprite(n)
        print(f"    {n!r:<28} → front={r.front.split('/')[-1]}")

    sys.exit(0 if result["missing"] == 0 else 1)
