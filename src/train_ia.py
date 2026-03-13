import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback # Añadimos CallbackList y BaseCallback
from env.pokemon_env import PokemonEnv
import numpy as np

# --- NUEVO: Clase para trackear victorias en W&B ---
class PokeWinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PokeWinRateCallback, self).__init__(verbose)
        self.wins = []

    def _on_step(self) -> bool:
        # Buscamos 'is_win' en la info que devuelve tu PokemonEnv
        for info in self.locals['infos']:
            if 'is_win' in info:
                self.wins.append(1 if info['is_win'] else 0)
        
        # Cada 2000 pasos (aprox 2 combates largos), mandamos la media a W&B
        if self.n_calls % 2000 == 0 and len(self.wins) > 0:
            # Media de los últimos 50 combates para ver la tendencia
            win_rate = np.mean(self.wins[-50:]) 
            self.logger.record("env/win_rate", win_rate)
        return True
# --------------------------------------------------

def train():
    os.makedirs("models/best_model_s3/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)

    env = PokemonEnv()

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        n_steps=4096,      # <--- Mayor horizonte
        batch_size=128,
        n_epochs=10,
        ent_coef=0.01,     # <--- Fomenta la exploración
        gamma=0.99,
        device="auto" 
    )
    # 3. Configuración de Callbacks
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/best_model_s3/',
        log_path='./logs/', 
        eval_freq=10000,
        deterministic=True
    )
    
    # Añadimos nuestro nuevo contador de victorias
    win_rate_callback = PokeWinRateCallback()

    # Combinamos ambos callbacks
    callback_list = CallbackList([eval_callback, win_rate_callback])

    print("\n🚀 Iniciando entrenamiento NIVEL AVANZADO...")
    print("Reglas: Niebla de guerra, Penalización por cambio y Rival Inteligente.\n")

    # 4. Entrenamiento con la lista de callbacks
    model.learn(
        total_timesteps=200000, 
        callback=callback_list, # <--- Usamos la lista aquí
        progress_bar=True
    )

    model.save("models/pokemon_ia_v5_avanzada")
    print("\n✅ Entrenamiento completado. Modelo 'v5' listo.")

if __name__ == "__main__":
    train()