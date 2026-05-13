"""
etl_process.py
~~~~~~~~~~~~~~
ETL pipeline: PokeAPI → Transform → SQLite (pokemon_bigdata.db).

Fetches base stats and typing for the first 151 Pokémon (Generation I)
from the public PokeAPI, transforms the JSON responses into flat rows,
and inserts them into the ``pokemon_stats`` table via INSERT OR REPLACE
(idempotent — safe to re-run).

The database schema is guaranteed to exist before any insert thanks to
``database_manager.init_db()``, which is called once at the start of the
pipeline run.

Usage
-----
    python etl_process.py           # fetch all 151 Pokémon
    python etl_process.py --limit 20  # quick test with the first 20
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from typing import Optional

import requests

try:
    from database_manager import init_db
except ImportError:
    from database_manager import init_db  # noqa: F811  (same — covered above)

# ── Configuration ─────────────────────────────────────────────────────────────

POKEAPI_BASE    = "https://pokeapi.co/api/v2/pokemon"
DB_PATH         = "pokemon_bigdata.db"
KANTO_LIMIT     = 151
COMMIT_EVERY    = 10       # rows between intermediate commits
REQUEST_TIMEOUT = 10       # seconds per HTTP request
RETRY_ATTEMPTS  = 3        # retries on transient network errors
RETRY_BACKOFF   = 2.0      # seconds to wait between retries (doubles each attempt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Core functions ─────────────────────────────────────────────────────────────


def _fetch_pokemon(pokemon_id: int) -> Optional[dict]:
    """
    Fetch raw JSON for a single Pokémon from PokeAPI with retries.

    Parameters
    ----------
    pokemon_id : int
        National Pokédex number (1–151 for Kanto).

    Returns
    -------
    dict | None
        Parsed JSON response, or None if all retry attempts failed.
    """
    url = f"{POKEAPI_BASE}/{pokemon_id}"
    wait = RETRY_BACKOFF
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt < RETRY_ATTEMPTS:
                log.warning(
                    "ID %d — attempt %d/%d failed (%s). Retrying in %.1fs…",
                    pokemon_id, attempt, RETRY_ATTEMPTS, exc, wait,
                )
                time.sleep(wait)
                wait *= 2
            else:
                log.error("ID %d — all %d attempts failed: %s", pokemon_id, RETRY_ATTEMPTS, exc)
    return None


def _transform(raw: dict) -> tuple:
    """
    Extract a flat row from a raw PokeAPI Pokémon response.

    Parameters
    ----------
    raw : dict
        The full JSON object returned by /api/v2/pokemon/{id}.

    Returns
    -------
    tuple
        ``(id, name, hp, attack, defense, sp_atk, sp_def, speed, type1, type2)``
        ready for parameterised SQL insertion.
    """
    stats = {s["stat"]["name"]: s["base_stat"] for s in raw["stats"]}
    types = [t["type"]["name"] for t in raw["types"]]

    return (
        raw["id"],
        raw["name"].capitalize(),
        stats["hp"],
        stats["attack"],
        stats["defense"],
        stats["special-attack"],
        stats["special-defense"],
        stats["speed"],
        types[0],
        types[1] if len(types) > 1 else None,
    )


_INSERT_SQL = """
    INSERT OR REPLACE INTO pokemon_stats
        (id, name, hp, attack, defense, sp_attack, sp_defense, speed, type1, type2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def fetch_and_store_pokemon(limit: int = KANTO_LIMIT) -> None:
    """
    Run the full ETL pipeline for the first ``limit`` Pokémon.

    Parameters
    ----------
    limit : int
        Number of Pokémon to fetch (default: 151, all of Kanto).
    """
    # ── E: Extract — ensure schema exists before any writes ─────────────────
    init_db()

    log.info("🚀 Starting ETL pipeline  (limit=%d, db=%s)", limit, DB_PATH)
    inserted = skipped = 0

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for pokemon_id in range(1, limit + 1):
            raw = _fetch_pokemon(pokemon_id)
            if raw is None:
                skipped += 1
                continue

            # ── T: Transform ─────────────────────────────────────────────────
            row = _transform(raw)

            # ── L: Load ──────────────────────────────────────────────────────
            cursor.execute(_INSERT_SQL, row)
            inserted += 1

            if inserted % COMMIT_EVERY == 0:
                conn.commit()
                log.info("  ✅ %d Pokémon processed…", inserted)

        conn.commit()

    log.info(
        "🏁 ETL complete — %d inserted, %d skipped (errors). DB: %s",
        inserted, skipped, DB_PATH,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL pipeline: PokeAPI → SQLite (pokemon_bigdata.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=KANTO_LIMIT,
        help=f"Number of Pokémon to fetch (default: {KANTO_LIMIT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_and_store_pokemon(limit=args.limit)
