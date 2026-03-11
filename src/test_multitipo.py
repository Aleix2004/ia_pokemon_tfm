import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Configurar rutas para que encuentre la carpeta 'env'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.pokemon_env import PokemonEnv 

def run_test():
    # 2. Cargar el modelo (asegúrate de que la ruta sea correcta)
    model_path = "models/pokemon_ia_v3_multitipo"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: No se encuentra el modelo en {model_path}.zip")
        return

    model = PPO.load(model_path)
    env = PokemonEnv()

    rivales = ["Charmander", "Squirtle", "Bulbasaur", "Pikachu"]
    resultados = {rival: {"victorias": 0, "partidas": 10} for rival in rivales}

    print("\n--- INICIANDO TEST DE ESTRÉS MULTITIPO ---")

    for rival_nombre in rivales:
        print(f"\nProbando contra: {rival_nombre.upper()}")
        for i in range(10): # 10 partidas por rival para un test rápido
            # Intentamos forzar al rival si tu reset lo permite
            obs, info = env.reset(options={"rival": rival_nombre}) 
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # .item() para evitar el error de numpy array
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                total_reward = reward # Guardamos la recompensa final del combate

            # Lógica de victoria: si la recompensa final es positiva, contamos victoria
            if total_reward > 0:
                resultados[rival_nombre]["victorias"] += 1
            
            print(f"  Partida {i+1}: Recompensa = {total_reward} | {'GANADO' if total_reward > 0 else 'PERDIDO'}")

    # 3. Mostrar Resumen Final
    print("\n" + "="*30)
    print("      RESULTADOS FINALES")
    print("="*30)
    for rival, data in resultados.items():
        win_rate = (data["victorias"] / data["partidas"]) * 100
        print(f"{rival.ljust(12)} | Win Rate: {win_rate:>5}% ({data['victorias']}/{data['partidas']})")
    print("="*30)

if __name__ == "__main__":
    run_test()