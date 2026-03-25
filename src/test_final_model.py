import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from env.pokemon_env import PokemonEnv

def heuristic_policy(env):
    """
    Simula un oponente con lógica básica: 
    Elige el movimiento que más daño hace basándose en la tabla de tipos.
    """
    # En un entorno real, aquí accederías a la lógica de daño del motor.
    # Para este test, si no tienes acceso directo, el bot elegirá una acción aleatoria
    # o puedes definir acciones fijas.
    return env.action_space.sample() 

def evaluate_vs_bot(model_path, num_games=100):
    env = PokemonEnv()
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: No se encuentra el modelo en {model_path}")
        return

    model = PPO.load(model_path)
    victorias_ia = 0

    print(f"\n--- Iniciando Test: IA vs Heuristic Bot ({num_games} partidas) ---")

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            # Tu IA decide (sin aleatoriedad para ver su máximo nivel)
            action_ia, _ = model.predict(obs, deterministic=True)
            
            # El Bot de reglas fijas decide
            # Nota: Si tu env.step permite pasar una acción específica para el rival:
            action_rival = heuristic_policy(env)
            
            obs, reward, done, _, info = env.step(action_ia, action_rival=action_rival)
            
            if done:
                if info.get('is_win', False):
                    victorias_ia += 1

        if (i + 1) % 10 == 0:
            print(f"Partidas jugadas: {i+1}/{num_games} - Win Rate actual: {victorias_ia/(i+1):.2f}")

    win_rate = victorias_ia / num_games
    print(f"\nRESULTADO FINAL:")
    print(f"Win Rate de la IA: {win_rate * 100:.2f}%")
    
    if win_rate > 0.8:
        print("Estado: NIVEL ÉLITE (La IA domina las mecánicas)")
    elif win_rate > 0.5:
        print("Estado: NIVEL COMPETITIVO (La IA es sólida)")
    else:
        print("Estado: NECESITA MEJORAR (Falta generalización)")

if __name__ == "__main__":
    # Cambia esta ruta por la de tu mejor modelo cuando acabe
    PATH_MODELO = "models/best_self_play/model_final_v4"
    evaluate_vs_bot(PATH_MODELO)