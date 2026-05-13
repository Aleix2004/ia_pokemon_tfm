"""
database_manager.py
~~~~~~~~~~~~~~~~~~~
SQLite schema initialisation for the Pokémon IA project.

Defines and creates all three core tables:

* ``pokemon_stats``  — ETL output (populated by etl_process.py)
* ``battle_logs``    — high-level battle history (winner, turns)
* ``v_logs``         — per-turn battle log used by the dashboard

All CREATE statements use ``IF NOT EXISTS``, making this module safe to
call multiple times (idempotent).  The connection context manager ensures
the database is properly closed even if an exception occurs.

Usage
-----
    # Standalone: create the schema from the command line
    python database_manager.py

    # Programmatic: import and call at application start
    from database_manager import init_db
    init_db()
"""
from __future__ import annotations

import logging
import sqlite3

DB_PATH = "pokemon_bigdata.db"

log = logging.getLogger(__name__)

# ── DDL statements ─────────────────────────────────────────────────────────────

_DDL_POKEMON_STATS = """
    CREATE TABLE IF NOT EXISTS pokemon_stats (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        type1       TEXT,
        type2       TEXT,
        hp          INTEGER,
        attack      INTEGER,
        defense     INTEGER,
        sp_attack   INTEGER,
        sp_defense  INTEGER,
        speed       INTEGER
    )
"""

_DDL_BATTLE_LOGS = """
    CREATE TABLE IF NOT EXISTS battle_logs (
        id        INTEGER  PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        winner    TEXT,
        turns     INTEGER
    )
"""

_DDL_V_LOGS = """
    CREATE TABLE IF NOT EXISTS v_logs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        ia_move_name        TEXT,
        rival_move          TEXT,
        ia_move_type        TEXT,
        rival_move_type     TEXT,
        ia_effectiveness    TEXT,
        rival_effectiveness TEXT,
        hp_ia               REAL,
        hp_rival            REAL,
        reward              REAL
    )
"""

_ALL_DDL = [_DDL_POKEMON_STATS, _DDL_BATTLE_LOGS, _DDL_V_LOGS]


# ── Public API ─────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """
    Create all project tables in *db_path* if they do not already exist.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Created automatically if missing.
    """
    with sqlite3.connect(db_path) as conn:
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()
    log.info("✅ Database ready: %s", db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    init_db()
