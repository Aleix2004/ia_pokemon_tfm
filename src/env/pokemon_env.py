import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# --- DICCIONARIOS DE APOYO ---
NOMBRES_TIPOS = {0: "Fuego", 1: "Agua", 2: "Planta", 3: "Eléctrico"}
NOMBRES_POKEMONS = {
    0: "Charmander", 
    1: "Squirtle",   
    2: "Bulbasaur",  
    3: "Pikachu"     
}

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        
        # 4 acciones: 0: Fuego, 1: Agua, 2: Planta, 3: Eléctrico
        self.action_space = spaces.Discrete(4)
        
        # Observación: [Vida propia, Vida rival, Tipo Rival (normalizado), Aux]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Lógica de efectividad de tipos
        self.tabla_tipos = {
            0: {2: 2.0, 1: 0.5}, # Fuego > Planta, < Agua
            1: {0: 2.0, 2: 0.5}, # Agua > Fuego, < Planta
            2: {1: 2.0, 0: 0.5}, # Planta > Agua, < Fuego
            3: {1: 2.0}          # Eléctrico > Agua
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # La IA siempre será Charizard
        self.mi_nombre = "Charizard"
        
        # Elegimos un ID de rival (0 a 3) de forma aleatoria
        self.rival_id = random.randint(0, 3)
        self.nombre_rival = NOMBRES_POKEMONS[self.rival_id]
        self.tipo_rival_nombre = NOMBRES_TIPOS[self.rival_id]

        # Normalizamos el tipo para la red neuronal
        tipo_rival_norm = self.rival_id / 3.0 
        
        # Estado inicial
        self.state = np.array([1.0, 1.0, tipo_rival_norm, 0.5], dtype=np.float32)
        
        # Info extra para el script de visualización
        info = {
            "mi_nombre": self.mi_nombre,
            "nombre_rival": self.nombre_rival,
            "tipo_rival": self.tipo_rival_nombre
        }
        
        return self.state, info

    def step(self, action):
        tipo_rival_actual = int(round(self.state[2] * 3))
        
        multiplicador = self.tabla_tipos.get(action, {}).get(tipo_rival_actual, 1.0)
        daño_base = 0.15
        daño_final = daño_base * multiplicador
        
        # Aplicar daño
        self.state[1] -= daño_final   
        self.state[0] -= 0.10          
        
        self.state = np.clip(self.state, 0, 1)

        reward = daño_final * 10 
        
        terminated = False
        if self.state[1] <= 0:   
            reward += 20.0
            terminated = True
        elif self.state[0] <= 0: 
            reward -= 10.0
            terminated = True
            
        info = {
            "daño_realizado": daño_final,
            "multiplicador": multiplicador
        }
            
        return self.state, reward, terminated, False, info