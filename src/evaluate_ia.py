import gymnasium as gym  # Cambio clave: gymnasium
import numpy as np
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer

# IMPORTANTE: Importa aquí tu clase del entorno para que se registre
# Si tu clase se llama PokemonEnv y está en env_setup.py:
# from src.env_setup import PokemonEnv 

def evaluate():
    # 1. Cargar el modelo
    model = PPO.load("ppo_pokemon_model") 
    
    # 2. Definir el rival
    rival = RandomPlayer(battle_format="gen8randombattle")
    
    # 3. Crear el entorno usando Gymnasium
    # Asegúrate de que el nombre coincide con el que usaste en el entrenamiento
    try:
        env = gym.make("pkmn_env", opponent=rival)
    except gym.error.NameError:
        print("Error: El entorno 'pkmn_env' no está registrado.")
        print("Asegúrate de importar la clase que define el entorno en este script.")
        return

    n_episodes = 100
    wins = 0

    print(f"--- Evaluando IA en {n_episodes} combates ---")

    for i in range(n_episodes):
        obs, info = env.reset() # Gymnasium devuelve (obs, info)
        done = False
        truncated = False # Gymnasium usa 'truncated' para límites de tiempo
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action) # Step de 5 valores
            
            if (done or truncated) and reward > 0: 
                wins += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progreso: {i + 1}/{n_episodes}...")

    win_rate = (wins / n_episodes) * 100
    print(f"\nRESULTADO: {wins}% de victorias (Win Rate: {win_rate}%)")

if __name__ == "__main__":
    evaluate()