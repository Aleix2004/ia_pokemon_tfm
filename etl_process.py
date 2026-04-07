import requests
import sqlite3
import time

def fetch_and_store_pokemon():
    conn = sqlite3.connect('pokemon_bigdata.db')
    cursor = conn.cursor()
    
    print("🚀 Iniciando Ingesta de Datos (ETL)...")
    
    # Vamos a por los primeros 151 (Kanto) para no tardar mucho ahora
    for i in range(1, 152):
        try:
            r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}").json()
            
            # Extraer stats
            stats = {s['stat']['name']: s['base_stat'] for s in r['stats']}
            types = [t['type']['name'] for t in r['types']]
            type1 = types[0]
            type2 = types[1] if len(types) > 1 else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO pokemon_stats 
                (id, name, hp, attack, defense, special_attack, special_defense, speed, type1, type2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r['id'], r['name'].capitalize(),
                stats['hp'], stats['attack'], stats['defense'],
                stats['special-attack'], stats['special-defense'], stats['speed'],
                type1, type2
            ))
            
            if i % 10 == 0:
                print(f"✅ Procesados {i} Pokémon...")
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error en ID {i}: {e}")

    conn.commit()
    conn.close()
    print("🏁 Ingesta completada. ¡Ya tienes tu Dataset SQL de Big Data!")

if __name__ == "__main__":
    fetch_and_store_pokemon()