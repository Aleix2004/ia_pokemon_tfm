import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import wandb
from wandb.integration.sb3 import WandbCallback

# Asegurar que Python encuentra la carpeta env
sys.path.append(os.getcwd())
from env.pokemon_env import PokemonEnv

def train():
    # 1. Configuración de W&B para la Semana 5
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100000, 
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.05, # Subimos un poco la exploración para que pruebe la Parálisis
    }

    run = wandb.init(
        project="ia-pokemon-tfm",
        name="semana-5-paralisis-inicio",
        config=config,
        sync_tensorboard=True,
    )

    # 2. Instanciar entorno y modelo desde cero
    env = PokemonEnv()
    
    # IMPORTANTE: No cargamos el anterior porque el observation_space ha cambiado
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        ent_coef=config["ent_coef"],
        tensorboard_log="./logs/"
    )

    # 3. Entrenamiento
    print("🚀 Entrenando Semana 5: Introduciendo mecánica de Parálisis...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_path="models/semana_5/",
            verbose=2
        )
    )

    # 4. Guardar
    os.makedirs("models/best_model_s5", exist_ok=True)
    model.save("models/best_model_s5/best_model")
    print("✅ Entrenamiento completado. Nuevo modelo guardado en models/best_model_s5/")
    
    wandb.finish()

if __name__ == "__main__":
    train()