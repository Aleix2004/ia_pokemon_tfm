"""
check_data.py
~~~~~~~~~~~~~
Quick sanity-check script for the ``pokemon_stats`` ETL table.

Runs two summary queries against ``pokemon_bigdata.db`` and prints the
results to stdout.  Useful after running ``etl_process.py`` to confirm the
data landed correctly.

Usage
-----
    python check_data.py
"""
from __future__ import annotations

import sqlite3

import pandas as pd

DB_PATH = "pokemon_bigdata.db"

_QUERY_BY_TYPE = """
    SELECT   type1,
             COUNT(*) AS cantidad
    FROM     pokemon_stats
    GROUP BY type1
    ORDER BY cantidad DESC
"""

_QUERY_TOP_ATTACKERS = """
    SELECT   name,
             attack
    FROM     pokemon_stats
    ORDER BY attack DESC
    LIMIT    5
"""


def check_data(db_path: str = DB_PATH) -> None:
    """
    Print a brief statistical overview of the ETL dataset.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """
    with sqlite3.connect(db_path) as conn:
        df_types = pd.read_sql_query(_QUERY_BY_TYPE, conn)
        df_top   = pd.read_sql_query(_QUERY_TOP_ATTACKERS, conn)

    print("📊 Pokémon count by primary type:")
    print(df_types.to_string(index=False))

    print("\n⚔️  Top 5 physical attackers:")
    print(df_top.to_string(index=False))


if __name__ == "__main__":
    check_data()
