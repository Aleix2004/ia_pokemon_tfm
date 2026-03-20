import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE RUTA MANUAL ---
# Usamos la ruta completa que veo en tu terminal
MODEL_FOLDER = r"C:\Users\Alumno\Desktop\ia_pokemon_tfm\models"
MODEL_NAME = "pokemon_ia_v5_avanzada"  # SIN el .zip al final
FULL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

print(f"🔍 Intentando cargar desde: {FULL_PATH}.zip")

try:
    if os.path.exists(FULL_PATH + ".zip"):
        model = PPO.load(FULL_PATH)
        print("✅ ¡CEREBRO CARGADO CORRECTAMENTE!")
    else:
        print(f"❌ ERROR: No veo el archivo en {FULL_PATH}.zip")
        model = None
except Exception as e:
    print(f"⚠️ Error al cargar: {e}")
    model = None

class BattleState(BaseModel):
    ia_hp: int
    rival_hp: int

@app.post("/predict")
async def predict(state: BattleState):
    if model is None:
        return {"move_name": "Thunderbolt", "error": "Modelo no cargado"}
    
    obs = np.array([state.ia_hp, state.rival_hp], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    movimientos = ["Thunderbolt", "Iron Tail", "Quick Attack", "Thunder Wave"]
    
    return {"move_name": movimientos[int(action)]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)