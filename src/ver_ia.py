import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env.pokemon_env import PokemonEnv

# Mapeo de ataques de la IA
NOMBRES_ATAQUES = {
    0: "Llamarada (Fuego)",
    1: "Hidrobomba (Agua)",
    2: "Rayo Solar (Planta)",
    3: "Trueno (Eléctrico)"
}

def test():
    try:
        env = PokemonEnv()
    except Exception as e:
        print(f"❌ Error al inicializar el entorno: {e}")
        return

    print("\n🧠 --- Cargando el cerebro de la IA ---")
    modelo_path = "models/pokemon_ia_v2_final"
    try:
        model = PPO.load(modelo_path)
        print(f"✅ Modelo '{modelo_path}' cargado.")
    except Exception as e:
        print(f"❌ No se encontró el modelo.")
        return

    # Reset y obtención de nombres
    obs, info = env.reset()
    mi_nombre = info.get("mi_nombre", "IA")
    nombre_rival = info.get("nombre_rival", "Pokémon Rival")
    tipo_rival = info.get("tipo_rival", "Desconocido")

    print(f"\n🌟 ¡Tu {mi_nombre} se enfrenta a un {nombre_rival} ({tipo_rival})!")
    print("🥊 --- INICIO DEL COMBATE --- 🥊")
    print("-" * 50)

    done = False
    total_reward = 0
    turno = 1

    while not done:
        action_array, _ = model.predict(obs, deterministic=True)
        action = action_array.item()

        obs, reward, terminated, truncated, info_step = env.step(action)
        done = terminated or truncated
        total_reward += reward

        ataque_nombre = NOMBRES_ATAQUES.get(action, f"Ataque {action}")
        
        print(f"Turno {turno}:")
        print(f"  🤖 {mi_nombre} usa: {ataque_nombre}")
        
        mult = info_step.get("multiplicador", 1.0)
        if mult > 1.0: print("  💥 ¡Es súper efectivo!")
        elif mult < 1.0: print("  🛡️ No es muy efectivo...")

        # Barras de vida visuales
        bar_ia = "=" * int(obs[0] * 20)
        bar_rv = "=" * int(obs[1] * 20)
        print(f"  🏥 HP {mi_nombre:10}: [{bar_ia:20}] {obs[0]*100:.0f}%")
        print(f"  🏥 HP {nombre_rival:10}: [{bar_rv:20}] {obs[1]*100:.0f}%")
        print("-" * 50)

        turno += 1

    print("\n--- 🏁 FIN DEL COMBATE 🏁 ---")
    if obs[1] <= 0:
        print(f"🏆 ¡Victoria! {mi_nombre} ha derrotado a {nombre_rival}.")
    else:
        print(f"💀 {mi_nombre} ha sido derrotado...")
    
    print(f"📊 Recompensa total: {total_reward:.2f}")

if __name__ == "__main__":
    test()