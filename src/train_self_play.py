import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import wandb
import numpy as np

# Asegurar acceso al entorno
sys.path.append(os.getcwd())
from env.pokemon_env import PokemonEnv

def calcular_expected_score(r_aprendiz, r_maestro):
    """Calcula la probabilidad de victoria del aprendiz según el sistema ELO"""
    return 1 / (1 + 10 ** ((r_maestro - r_aprendiz) / 400))

def train():
    env = PokemonEnv()
    
    # 1. Cargar el modelo de la Semana 5 como base inicial
    maestro_path = "models/best_model_s5/best_model"
    if not os.path.exists(maestro_path + ".zip"):
        print(f"❌ Error: No se encuentra el modelo base en {maestro_path}")
        return

    print("📜 Cargando modelos iniciales...")
    maestro = PPO.load(maestro_path)
    aprendiz = PPO.load(maestro_path, env=env)
    
    # 2. Configuración de Weights & Biases
    wandb.init(
        project="ia-pokemon-tfm", 
        name="self-play-elo-evolution",
        config={
            "k_factor": 32,
            "generaciones": 5,
            "timesteps_per_gen": 20000,
            "eval_matches": 50
        }
    )
    
    elo_aprendiz = 1200
    elo_maestro = 1200
    k_factor = 32

    # 

    for generacion in range(1, 6):
        print(f"\n--- 🚀 GENERACIÓN {generacion} ---")
        
        # A. Entrenar al aprendiz contra el bot interno (base)
        print(f"🏋️ Entrenando aprendiz ({generacion})...")
        aprendiz.learn(total_timesteps=20000, reset_num_timesteps=False)
        
        # B. Torneo de Validación: Aprendiz vs Maestro
        print(f"⚔️ Torneo de Validación: Aprendiz (ELO {elo_aprendiz:.0f}) vs Maestro (ELO {elo_maestro:.0f})...")
        victorias_aprendiz = 0
        combates_eval = 50
        
        for _ in range(combates_eval):
            obs, _ = env.reset()
            done = False
            while not done:
                # Acción del aprendiz (Perspectiva normal)
                action_ia, _ = aprendiz.predict(obs, deterministic=True)
                
                # Acción del maestro (Perspectiva invertida)
                obs_rival = env._get_obs(for_rival=True)
                action_rival, _ = maestro.predict(obs_rival, deterministic=True)
                
                # Paso en el entorno con ambas acciones
                obs, _, done, _, info = env.step(action_ia, action_rival=action_rival)
                
                if done and info['is_win']:
                    victorias_aprendiz += 1
        
        # C. Cálculo y actualización de ELO
        win_rate = victorias_aprendiz / combates_eval
        expected_a = calcular_expected_score(elo_aprendiz, elo_maestro)
        
        # El ELO del aprendiz sube o baja según su desempeño contra el maestro
        elo_aprendiz += k_factor * (win_rate - expected_a)
        
        print(f"📊 Resultado Gen {generacion}: Win Rate {win_rate:.2f} | Nuevo ELO Aprendiz: {elo_aprendiz:.2f}")
        
        wandb.log({
            "generacion": generacion,
            "elo_rating": elo_aprendiz,
            "win_rate_vs_maestro": win_rate,
            "victorias": victorias_aprendiz
        })

        # D. Evolución: Si el aprendiz es mejor, se convierte en el nuevo Maestro
        if win_rate > 0.55:
            print("🏆 ¡Evolución detectada! El aprendiz ahora es el maestro.")
            path_gen = f"models/self_play_gen_{generacion}"
            aprendiz.save(path_gen)
            maestro = PPO.load(path_gen)
            # El ELO del maestro se actualiza al nivel alcanzado por el aprendiz
            elo_maestro = elo_aprendiz

    # 3. Guardar modelo final
    os.makedirs("models/best_self_play", exist_ok=True)
    aprendiz.save("models/best_self_play/model_final")
    print("\n✅ Entrenamiento de Self-Play completado.")
    wandb.finish()

if __name__ == "__main__":
    train()