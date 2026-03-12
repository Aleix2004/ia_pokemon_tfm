import gymnasium as gym
from stable_baselines3 import PPO
from env.pokemon_env import PokemonEnv
import os

def batalla_visual():
    env = PokemonEnv()
    pokedex = {
        0: {"nombre": "Charizard", "emoji": "🔥"},
        1: {"nombre": "Blastoise", "emoji": "💧"},
        2: {"nombre": "Venusaur", "emoji": "🌿"},
        3: {"nombre": "Pikachu", "emoji": "⚡"}
    }

    print("\n🧠 --- Cargando el cerebro de la IA ---")
    model = PPO.load("models/pokemon_ia_v3_multitipo")

    obs, info = env.reset()
    done = False
    
    # Variables para rastrear tipos actuales
    t_ia = info["tipo_ia"]
    t_hum = info["tipo_rival"]

    print(f"\n      ¡COMIENZA EL DUELO ESTRATÉGICO!      ")
    print("-" * 45)

    while not done:
        # 1. IA decide su movimiento
        action_ia, _ = model.predict(obs, deterministic=True)
        action_ia = int(action_ia)

        # 2. Interfaz de usuario
        pk_ia = pokedex[t_ia]
        pk_hum = pokedex[t_hum]
        
        print(f"\nSITUACIÓN ACTUAL:")
        print(f"🤖 IA: {pk_ia['nombre']} {pk_ia['emoji']} | 👤 TÚ: {pk_hum['nombre']} {pk_hum['emoji']}")
        print(f"¿Qué hará {pk_hum['nombre']}?")
        print(f"--- ATACAR con {pk_hum['emoji']}: Pulsa {t_hum}")
        print(f"--- CAMBIAR Pokémon: Elige 0, 1, 2 o 3 (diferente al actual)")
        
        try:
            accion_usuario = int(input("Tu decisión: "))
        except:
            accion_usuario = t_hum

        # 3. Ejecutar turno simultáneo
        obs, reward, terminated, truncated, info = env.step(action_ia, accion_usuario)
        
        # Actualizar tipos tras el turno
        t_ia = info["tipo_ia"]
        t_hum = info["tipo_rival"]

        # 4. Mostrar eventos y barras
        for evento in info["eventos"]:
            print(f"📢 {evento}")

        def barra(hp): return "█" * int(hp * 20) + "-" * (20 - int(hp * 20))
        
        print(f"\n{pokedex[t_ia]['nombre']} (IA)  : [{barra(obs[0])}] {int(obs[0]*100)}%")
        print(f"{pokedex[t_hum]['nombre']} (TÚ) : [{barra(obs[1])}] {int(obs[1]*100)}%")
        
        done = terminated or truncated

    print("\n" + "="*45)
    if obs[1] <= 0:
        print("💀 ¡Tu equipo ha sido derrotado! La IA es superior.")
    else:
        print("🏆 ¡INCREÍBLE! Has derrotado a la Inteligencia Artificial.")
    print("="*45)

if __name__ == "__main__":
    batalla_visual()