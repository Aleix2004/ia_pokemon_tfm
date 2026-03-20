import requests
import os

def download_pokemon_assets(pokemon_list):
    base_url = "https://play.pokemonshowdown.com/sprites/ani"
    back_url = "https://play.pokemonshowdown.com/sprites/ani-back"
    target_dir = "web_engine/frontend/assets/pokemon"
    
    os.makedirs(target_dir, exist_ok=True)

    for p in pokemon_list:
        p = p.lower()
        # Descargar frente (rival)
        f_res = requests.get(f"{base_url}/{p}.gif")
        if f_res.status_code == 200:
            with open(f"{target_dir}/{p}_front.gif", "wb") as f:
                f.write(f_res.content)
            print(f"✅ {p} frente descargado")

        # Descargar espalda (tu IA)
        b_res = requests.get(f"{back_url}/{p}.gif")
        if b_res.status_code == 200:
            with open(f"{target_dir}/{p}_back.gif", "wb") as f:
                f.write(b_res.content)
            print(f"✅ {p} espalda descargado")

if __name__ == "__main__":
    # Añade aquí los que quieras usar
    mis_pokemon = ["raichu", "dragonite", "charizard", "blastoise"]
    download_pokemon_assets(mis_pokemon)