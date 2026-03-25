import gymnasium as gym
import wandb
import os
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from env.pokemon_env import PokemonEnv

def main():
    # 1. Configuración de hiperparámetros (Lo que el algoritmo usará para aprender)
    config = {
        "policy_type": "MlpPolicy",       # Red neuronal estándar
        "total_timesteps": 50000,        # Cuántos turnos de práctica hará la IA
        "env_name": "Pokemon-IA-v2",
        "learning_rate": 0.0003,          # Velocidad de aprendizaje
    }

    # 2. Iniciar sesión en WandB
    run = wandb.init(
        project="pokemon-ia-tfm",
        name="entrenamiento-ppo-semana2",
        config=config,
        sync_tensorboard=True, # Sincroniza las métricas internas del algoritmo
    )

    # 3. Instanciar el entorno
    env = PokemonEnv()

    # 4. Definir el Cerebro (PPO - Proximal Policy Optimization)
    # verbose=1 nos dará una tabla de progreso en la terminal
    model = PPO(
        config["policy_type"], 
        env, 
        learning_rate=config["learning_rate"],
        verbose=1, 
        tensorboard_log=f"runs/{run.id}"
    )

    # 5. Entrenar con el "WandbCallback"
    # Esto enviará automáticamente las métricas avanzadas (loss, reward mean, etc.)
    print("--- Iniciando Entrenamiento de la IA ---")
    # Cambia esto en tu main.py:
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2,  # Quitamos la línea que daba error
        )
    )

    # 6. Guardar el modelo entrenado para no perderlo
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/pokemon_ia_v2_final")

    print("\n¡Entrenamiento finalizado! Revisa las curvas de aprendizaje en wandb.ai")
    run.finish()

if __name__ == "__main__":
    main()