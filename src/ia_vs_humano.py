import gymnasium as gym
from stable_baselines3 import PPO
from env.pokemon_env import PokemonEnv

def batalla_manual():
    env = PokemonEnv()
    
    print("\n🧠 --- Cargando el cerebro de la IA ---")
    try:
        model = PPO.load("models/pokemon_ia_v2_final")
        print("✅ IA lista para el combate.")
    except:
        print("❌ No se encontró el modelo.")
        return

    obs, info = env.reset()
    done = False
    turno = 1

    print("\n🎮 --- MODO HUMANO: TÚ VS IA --- 🎮")
    print("Intenta ganar a la IA eligiendo los ataques del rival.")

    while not done:
        print(f"\n--- Turno {turno} ---")
        
        # 1. La IA predice su acción
        action_ia, _ = model.predict(obs, deterministic=True)
        action_ia = action_ia.item()

        # 2. Tú eliges la acción del RIVAL (el entorno debe permitir esto o lo simulamos)
        # Nota: Como tu entorno actual es autogestionado, aquí vamos a ver 
        # cómo responde la IA a la situación actual.
        
        print(f"La IA ha decidido usar: Ataque {action_ia}")
        
        # Ejecutamos el turno
        obs, reward, terminated, truncated, info = env.step(action_ia)
        done = terminated or truncated
        
        print(f"Resultado del turno: Recompensa para IA: {reward}")
        turno += 1

    print("\n--- ¡FIN DEL COMBATE! ---")

if __name__ == "__main__":
    batalla_manual()