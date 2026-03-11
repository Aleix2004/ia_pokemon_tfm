import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os

# 1. Configurar rutas relativas
# Estamos en /src, subimos uno para ir a la raíz y luego a /models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ruta al archivo .zip (sin la extensión .zip para PPO.load)
PATH_MODELO = os.path.join(BASE_DIR, "models", "pokemon_ia_v3_multitipo")

# Añadir /src al path para encontrar la carpeta 'env'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from env.pokemon_env import PokemonEnv 
except ImportError:
    print("❌ No se encontró 'env.pokemon_env'. Revisa la estructura de carpetas.")
    sys.exit()

# 2. Inicializar Entorno
try:
    env = PokemonEnv(render_mode="human")
except TypeError:
    print("⚠️ El entorno no acepta render_mode. Usando modo normal.")
    env = PokemonEnv()

# 3. Cargar el Modelo Específico
if os.path.exists(PATH_MODELO + ".zip"):
    model = PPO.load(PATH_MODELO)
    print(f"✅ Modelo v3_multitipo cargado con éxito.")
else:
    print(f"❌ No encontré el archivo: {PATH_MODELO}.zip")
    print("Prueba a listar de nuevo con 'ls ../models' para verificar el nombre.")
    sys.exit()

# 4. Simulación
print("🚀 Iniciando combates...")
for i in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 1. La IA predice la acción
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. Convertimos la acción de Numpy a un entero simple
        action = int(action) # O action.item()
        
        # 3. Ejecutamos el paso en el entorno
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        
        try:
            env.render()
        except:
            pass
            
    print(f"🏆 Combate {i+1} terminado. Recompensa: {total_reward}")