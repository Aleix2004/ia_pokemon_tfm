import gym
import pokemon_env  # Tu entorno personalizado
from stable_baselines3 import PPO

def test_ia():
    # 1. Cargar el entorno
    env = gym.make('Pokemon-v0') 
    
    # 2. Cargar el "cerebro" que acabas de entrenar
    # Asegúrate de que el nombre coincide con el que guardaste
    model = PPO.load("pokemon_ia_model")
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    print("\n--- 🥊 INICIANDO COMBATE DE EXHIBICIÓN ---")
    
    while not done:
        # La IA elige la mejor acción según lo aprendido
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Mostrar qué ha pasado en este turno
        print(f"Turno {info.get('turno', 'N/A')}:")
        print(f" > IA usó: {info.get('ataque_ia', action)}")
        print(f" > Estado: Vida IA {info.get('hp_ia')} | Vida Rival {info.get('hp_rival')}")
        print("-" * 30)

    print(f"\n--- FIN DEL COMBATE ---")
    print(f"Resultado: {'🏆 VICTORIA' if total_reward > 0 else '💀 DERROTA'}")
    print(f"Recompensa total: {total_reward}")

if __name__ == "__main__":
    test_ia()