import sqlite3

def init_db():
    conn = sqlite3.connect('pokemon_bigdata.db')
    cursor = conn.cursor()
    
    # Tabla para el Dashboard (v_logs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS v_logs (
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
    ''')
    
    # Tabla histórica
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS battle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            winner TEXT,
            turns INTEGER
        )
    ''')

    # Dataset ETL (etl_process.py + Big Data section in dashboard)
    # This table was referenced by etl_process.py and dashboard.py but was
    # never created, causing OperationalError on every ETL run and dashboard load.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pokemon_stats (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            type1      TEXT,
            type2      TEXT,
            hp         INTEGER,
            attack     INTEGER,
            defense    INTEGER,
            sp_attack  INTEGER,
            sp_defense INTEGER,
            speed      INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Base de datos lista.")

if __name__ == "__main__":
    init_db()
