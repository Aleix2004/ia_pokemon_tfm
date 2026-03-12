import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from env.pokemon_env import PokemonEnv
import numpy as np

# --- WRAPPER PARA ENTRENAMIENTO ---
class TrainWrapper(gym.Wrapper):
    """
    Este wrapper hace que el rival tome acciones aleatorias para que 
    la IA tenga a alguien contra quien entrenar.
    """
    def step(self, action):
        # El rival elige una acción al azar (0, 1, 2 o 3)
        action_rival_aleatoria = self.env.action_space.sample()
        return self.env.step(action, action_rival_aleatoria)

def train():
    os.makedirs("models/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)

    # Instanciamos el entorno con el Wrapper
    base_env = PokemonEnv()
    env = TrainWrapper(base_env)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="auto" # Usa GPU si está disponible
    )

    # Callback para guardar el mejor modelo de la Semana 3
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/best_model_s3/',
        log_path='./logs/', 
        eval_freq=5000,
        deterministic=True
    )

    print("\n🚀 Iniciando entrenamiento Baseline (Semana 3)...")
    print("La IA aprenderá a combatir contra un rival que cambia y ataca al azar.\n")

    model.learn(
        total_timesteps=100000, 
        callback=eval_callback,
        progress_bar=True
    )

    # Guardamos como v4 porque incluye la lógica de CAMBIOS
    model.save("models/pokemon_ia_v4_con_cambios")
    print("\n✅ Semana 3 completada. Modelo 'v4' listo para evaluación.")

if __name__ == "__main__":
    train()