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
    
    conn.commit()
    conn.close()
    print("✅ Base de datos lista.")

if __name__ == "__main__":
    init_db()
