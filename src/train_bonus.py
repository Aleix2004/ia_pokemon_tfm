import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys
import wandb
from wandb.integration.sb3 import WandbCallback

# Asegurar rutas
sys.path.append(os.getcwd())
from env.pokemon_env import PokemonEnv

def train():
    # 1. Configuración de W&B
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 30000, # Un empujón extra de 30k pasos
        "learning_rate": 0.0002,   # Un poco más lento para no romper lo que ya sabe
        "ent_coef": 0.01          # Menos exploración, más ejecución de lo aprendido
    }

    run = wandb.init(
        project="ia-pokemon-tfm",
        name="bonus-refuerzo-antieléctrico",
        config=config,
        sync_tensorboard=True,
    )

    # 2. Preparar Entorno y Modelo
    env = PokemonEnv()
    model_path = "models/best_model_s3/best_model"

    if os.path.exists(model_path + ".zip"):
        print(f"🔄 Cargando modelo previo: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log="./logs/")
    else:
        print("🆕 No se encontró modelo previo. Empezando desde cero.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 3. Entrenamiento con el nuevo Reward
    print("🚀 Iniciando entrenamiento de refuerzo (Bonus Track)...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_path="models/bonus_model/",
            verbose=2
        )
    )

    # 4. Guardar resultado final
    os.makedirs("models/best_model_bonus", exist_ok=True)
    model.save("models/best_model_bonus/best_model")
    print("✅ Entrenamiento completado. Modelo guardado en models/best_model_bonus/")
    
    wandb.finish()

if __name__ == "__main__":
    train()