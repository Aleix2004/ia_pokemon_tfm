import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env.pokemon_env import PokemonEnv

def train():
    # 1. Crear carpetas para organizar el proyecto
    os.makedirs("models/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)

    # 2. Instanciar el entorno
    env = PokemonEnv()

    # 3. Configurar el modelo PPO
    # MLPPolicy es ideal para datos vectoriales (como la vida y el tipo)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        learning_rate=0.0003,  # Velocidad de aprendizaje equilibrada
        n_steps=2048,          # Pasos antes de actualizar la red
        batch_size=64,         # Tamaño del grupo de datos procesados
        n_epochs=10            # Veces que optimiza cada lote
    )

    # 4. Configurar Callbacks (Ahorran trabajo)
    # Guarda el mejor modelo automáticamente tras evaluar
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/best_model/',
        log_path='./logs/', 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )

    print("\n🚀 Iniciando entrenamiento multitipo...")
    print("La IA se enfrentará a Charmander, Squirtle, Bulbasaur y Pikachu aleatoriamente.\n")

    # 5. Entrenar (100.000 pasos es un buen número para empezar)
    model.learn(
        total_timesteps=100000, 
        callback=eval_callback,
        progress_bar=True
    )

    # 6. Guardar el modelo final
    model.save("models/pokemon_ia_v3_multitipo")
    print("\n✅ Entrenamiento completado. Modelo guardado como 'pokemon_ia_v3_multitipo'")

if __name__ == "__main__":
    train()