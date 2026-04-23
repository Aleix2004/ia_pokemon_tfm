import sqlite3
import pandas as pd

conn = sqlite3.connect('pokemon_bigdata.db')

# Consulta 1: ¿Cuántos Pokémon tenemos por cada tipo principal?
query = "SELECT type1, COUNT(*) as cantidad FROM pokemon_stats GROUP BY type1 ORDER BY cantidad DESC"
df = pd.read_sql_query(query, conn)

print("📊 CONTEO POR TIPO (Muestra de Big Data):")
print(df)

# Consulta 2: Los 5 más fuertes físicamente
query_top = "SELECT name, attack FROM pokemon_stats ORDER BY attack DESC LIMIT 5"
df_top = pd.read_sql_query(query_top, conn)

print("\n⚔️ TOP 5 ATACANTES FÍSICOS:")
print(df_top)

conn.close()