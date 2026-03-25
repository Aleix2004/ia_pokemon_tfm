import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import wandb
import numpy as np
import random

sys.path.append(os.getcwd())
from env.pokemon_env import PokemonEnv

def calcular_expected_score(r_a, r_m):
    return 1 / (1 + 10 ** ((r_m - r_a) / 400))

def get_random_past_model(models_dir, current_maestro_path):
    """Selecciona un modelo aleatorio de los guardados para diversificar el rival."""
    if not os.path.exists(models_dir):
        return current_maestro_path
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        return current_maestro_path
    
    selected = random.choice(model_files)
    return os.path.join(models_dir, selected.replace('.zip', ''))

def train():
    env = PokemonEnv()
    models_dir = "models/self_play_history"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("models/best_self_play", exist_ok=True)
    
    maestro_path = "models/best_model_s5/best_model"
    policy_kwargs = dict(net_arch=[256, 256])
    
    if os.path.exists(maestro_path + ".zip"):
        print(f"Cargando modelo previo: {maestro_path}")
        aprendiz = PPO.load(maestro_path, env=env, learning_rate=0.00008, ent_coef=0.05)
    else:
        print("⚠️ AVISO: No se encontró el modelo inicial. Creando uno nuevo...")
        aprendiz = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, ent_coef=0.05, policy_kwargs=policy_kwargs)

    rival_model = aprendiz
    wandb.init(project="ia-pokemon-tfm", name="self-play-v4-league")
    
    elo_aprendiz = 1321  
    elo_maestro = 1321
    k_factor = 24        
    
    for gen in range(1, 61):
        print(f"\n--- GENERACIÓN {gen} ---")
        aprendiz.learn(total_timesteps=100000, reset_num_timesteps=False)
        
        victorias = 0
        partidas_eval = 100
        for i in range(partidas_eval):
            if i % 10 == 0:
                past_model_path = get_random_past_model(models_dir, maestro_path)
                if os.path.exists(past_model_path + ".zip"):
                    rival_model = PPO.load(past_model_path)
                else:
                    rival_model = aprendiz 
            
            obs, _ = env.reset()
            done = False
            while not done:
                action_ia, _ = aprendiz.predict(obs, deterministic=True)
                obs_rival = env._get_obs(for_rival=True)
                action_rival, _ = rival_model.predict(obs_rival, deterministic=False)
                obs, _, done, _, info = env.step(action_ia, action_rival=action_rival)
                if done and info.get('is_win', False): 
                    victorias += 1
        
        wr = victorias / partidas_eval
        expected = calcular_expected_score(elo_aprendiz, elo_maestro)
        elo_aprendiz += k_factor * (wr - expected)
        
        wandb.log({"gen": gen, "elo": elo_aprendiz, "win_rate": wr})

        if wr > 0.55: 
            print(f"¡NUEVO MODELO EN LA LIGA! Elo: {elo_aprendiz:.2f}")
            path = os.path.join(models_dir, f"gen_{gen}_elo_{int(elo_aprendiz)}")
            aprendiz.save(path)
            maestro_path = path
            elo_maestro = elo_aprendiz
            aprendiz.learning_rate = max(aprendiz.learning_rate * 0.98, 0.00001) 

    # --- GUARDADO FINAL ---
    final_path_local = "models/best_self_play/model_final_v4"
    aprendiz.save(final_path_local)

    # 1. COPIA DE SEGURIDAD EN USB (UNIDAD E:)
    try:
        path_usb = "E:/model_final_v4"
        print(f"Salvando en USB (E:): {path_usb}")
        aprendiz.save(path_usb)
        print("¡GUARDADO EN USB EXITOSO!")
    except Exception as e:
        print(f"No se pudo guardar en USB: {e}")

    # 2. COPIA EN LA NUBE (WandB Artifacts)
    try:
        print("Subiendo modelo a la nube de WandB...")
        artifact = wandb.Artifact('modelo_final_v4', type='model')
        artifact.add_file(final_path_local + ".zip")
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"No se pudo subir a WandB: {e}")

    wandb.finish()

if __name__ == "__main__":
    train()