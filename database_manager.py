import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('pokemon_bigdata.db')
    cursor = conn.cursor()
    
    # Tabla 1: El Dataset de Pokémon (Punto 1 y 2 del PDF)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pokemon_stats (
            id INTEGER PRIMARY KEY,
            name TEXT,
            hp INTEGER,
            attack INTEGER,
            defense INTEGER,
            special_attack INTEGER,
            special_defense INTEGER,
            speed INTEGER,
            type1 TEXT,
            type2 TEXT
        )
    ''')
    
    # Tabla 2: Registro de batallas (Para el análisis de Big Data posterior)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS battle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            winner TEXT,
            turns INTEGER,
            ia_pokemon_left INTEGER,
            rival_pokemon_left INTEGER,
            total_damage_dealt FLOAT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Base de datos SQL inicializada correctamente.")

if __name__ == "__main__":
    init_db()