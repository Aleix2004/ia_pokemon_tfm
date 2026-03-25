import gymnasium as gym
import numpy as np
import os
import sys
import wandb
from stable_baselines3 import PPO

sys.path.append(os.getcwd()) 
from env.pokemon_env import PokemonEnv

def evaluate():
    # 1. Apuntamos al modelo de la Semana 5
    model_path = "models/best_model_s5/best_model" 
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: No se encuentra {model_path}.zip")
        return
    
    wandb.init(
        project="ia-pokemon-tfm", 
        name="evaluacion-semana-5-paralisis",
        config={"modelo": "PPO_S5_Paralisis", "n_combates": 100}
    )

    print(f"🔄 Cargando modelo con PARÁLISIS desde {model_path}...")
    model = PPO.load(model_path) 
    env = PokemonEnv()

    n_episodes = 100
    wins = 0
    paralisis_totales = 0
    derrotas_por_tipo = {} 

    print(f"\n--- ⚔️ EVALUANDO ESTRATEGIA DE ESTADO (S5) ---")

    for i in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action_ia, _ = model.predict(obs, deterministic=True)
            
            # Registrar si usa Onda Trueno (Acción 3)
            if action_ia == 3:
                paralisis_totales += 1
                
            obs, reward, terminated, truncated, info = env.step(int(action_ia))
            
            if terminated:
                if info.get('is_win'): 
                    wins += 1
                else:
                    t_rival = info.get('rival_type', 'Desconocido')
                    derrotas_por_tipo[t_rival] = derrotas_por_tipo.get(t_rival, 0) + 1
        
        if (i + 1) % 10 == 0:
            print(f"   > Combates evaluados: {i + 1}/{n_episodes}...")

    # --- LOGS FINALES ---
    win_rate = (wins / n_episodes) * 100
    avg_paralisis = paralisis_totales / n_episodes

    wandb.log({
        "win_rate": win_rate,
        "avg_paralisis_per_match": avg_paralisis,
        "total_wins": wins
    })

    print(f"\n========================================")
    print(f"🏆 WIN RATE S5: {win_rate:.2f}%")
    print(f"⚡ MEDIA ONDA TRUENO POR COMBATE: {avg_paralisis:.2f}")
    print(f"========================================")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    evaluate()