import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        # Obs: [Vida IA, Vida Rival, Tipo IA, Tipo Rival]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self.tabla_tipos = {
            0: {2: 2.0, 1: 0.5}, # Fuego
            1: {0: 2.0, 2: 0.5}, # Agua
            2: {1: 2.0, 0: 0.5}, # Planta
            3: {1: 2.0}          # Eléctrico
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tipo_ia = random.randint(0, 3)
        self.tipo_rival = random.randint(0, 3)
        self.hp_ia = 1.0
        self.hp_rival = 1.0
        
        self.state = np.array([self.hp_ia, self.hp_rival, self.tipo_ia/3.0, self.tipo_rival/3.0], dtype=np.float32)
        return self.state, {"tipo_ia": self.tipo_ia, "tipo_rival": self.tipo_rival}

    def step(self, action_ia, action_humano=None):
        # Si no hay acción humana (entrenamiento), el rival mantiene su tipo
        if action_humano is None: action_humano = self.tipo_rival

        ia_cambia = action_ia != self.tipo_ia
        humano_cambia = action_humano != self.tipo_rival
        
        recompensa_ia = 0
        eventos = []

        # --- FASE DE CAMBIOS ---
        if ia_cambia:
            self.tipo_ia = action_ia
            eventos.append(f"🔄 La IA ha cambiado su Pokémon.")
        
        if humano_cambia:
            self.tipo_rival = action_humano
            eventos.append(f"🔄 ¡Has cambiado tu Pokémon!")

        # --- FASE DE ATAQUE ---
        # Si un bando cambia, NO ataca ese turno (gasta el turno)
        
        # Ataque de la IA
        if not ia_cambia:
            mult = self.tabla_tipos.get(self.tipo_ia, {}).get(self.tipo_rival, 1.0)
            daño = 0.15 * mult
            self.hp_rival -= daño
            recompensa_ia += daño * 5
        
        # Ataque del Humano
        if not humano_cambia:
            mult = self.tabla_tipos.get(self.tipo_rival, {}).get(self.tipo_ia, 1.0)
            daño = 0.15 * mult
            self.hp_ia -= daño
            recompensa_ia -= daño * 5

        # Actualizar estado
        self.hp_ia = np.clip(self.hp_ia, 0, 1)
        self.hp_rival = np.clip(self.hp_rival, 0, 1)
        self.state = np.array([self.hp_ia, self.hp_rival, self.tipo_ia/3.0, self.tipo_rival/3.0], dtype=np.float32)

        terminated = self.hp_ia <= 0 or self.hp_rival <= 0
        return self.state, recompensa_ia, terminated, False, {"eventos": eventos, "tipo_ia": self.tipo_ia, "tipo_rival": self.tipo_rival}