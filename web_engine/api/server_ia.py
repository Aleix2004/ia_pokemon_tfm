import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

app = FastAPI()

# 1. Configuración de CORS para permitir que el Frontend (Docker) se comunique
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Ruta del modelo (Ajustada a tu estructura de carpetas)
MODEL_FOLDER = r"C:\Users\Alumno\Desktop\ia_pokemon_tfm\models"
MODEL_NAME = "pokemon_ia_v5_avanzada"
FULL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

print(f"\n🔍 Buscando modelo en: {FULL_PATH}.zip")

# 3. Carga del Modelo PPO
try:
    if os.path.exists(FULL_PATH + ".zip"):
        model = PPO.load(FULL_PATH)
        print("✅ ¡SISTEMA ONLINE: IA CARGADA CORRECTAMENTE!\n")
    else:
        print(f"❌ ERROR: No se encontró el archivo en {FULL_PATH}.zip")
        model = None
except Exception as e:
    print(f"⚠️ Error al cargar el modelo: {e}")
    model = None

# 4. Definición de la estructura de datos que recibe la API
class BattleState(BaseModel):
    ia_hp: int
    rival_hp: int

# 5. Punto de entrada para la predicción
@app.post("/predict")
async def predict(state: BattleState):
    if model is None:
        return {"move_name": "Quick Attack", "error": "Modelo no cargado en el servidor"}
    
    try:
        # CORRECCIÓN DE SHAPE: 
        # Tu modelo v5 espera 4 valores de entrada. 
        # Enviamos [HP_IA, HP_RIVAL, 0, 0] para cumplir con el requisito de la red neuronal.
        obs = np.array([state.ia_hp, state.rival_hp, 0, 0], dtype=np.float32)
        
        # Inferencia de la IA
        action, _ = model.predict(obs, deterministic=True)
        
        # Mapeo de los movimientos (deben coincidir con el entrenamiento)
        movimientos = ["Thunderbolt", "Iron Tail", "Quick Attack", "Thunder Wave"]
        
        idx = int(action)
        # Verificación de seguridad para el índice
        move_chosen = movimientos[idx] if idx < len(movimientos) else movimientos[0]
        
        print(f"🧠 IA analizó HP [{state.ia_hp}, {state.rival_hp}] -> Eligió: {move_chosen}")
        
        return {
            "move_name": move_chosen,
            "action_idx": idx
        }
        
    except Exception as e:
        print(f"❌ Error interno en la predicción: {e}")
        return {"move_name": "Quick Attack", "error": str(e)}

# 6. Ejecución del servidor
if __name__ == "__main__":
    import uvicorn
    # IMPORTANTE: host="0.0.0.0" permite que Docker (host.docker.internal) conecte con Windows
    uvicorn.run(app, host="0.0.0.0", port=8000)