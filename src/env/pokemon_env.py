import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        # Soportamos la observación de 5 parámetros que tienes definida
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.tabla_tipos = {
            0: {2: 2.0, 1: 0.5}, # Fuego
            1: {0: 2.0, 2: 0.5}, # Agua
            2: {1: 2.0, 0: 0.5}, # Planta
            3: {1: 2.0}          # Eléctrico
        }
        self.nombres_tipos = {0: "Fuego", 1: "Agua", 2: "Planta", 3: "Electrico"}
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tipo_ia = random.randint(0, 2)
        self.tipo_rival_real = random.randint(0, 3)
        self.tipo_rival_visible = -1.0 
        self.hp_ia = 1.0
        self.hp_rival = 1.0
        self.estado_rival = 0.0 
        self.estado_ia = 0.0    
        return self._get_obs(), {}

    def _get_obs(self, for_rival=False):
        if not for_rival:
            visible = self.tipo_rival_visible / 3.0 if self.tipo_rival_visible != -1.0 else 0.5
            return np.array([self.hp_ia, self.hp_rival, self.tipo_ia/2.0, visible, self.estado_rival], dtype=np.float32)
        else:
            return np.array([self.hp_rival, self.hp_ia, self.tipo_rival_real/3.0, self.tipo_ia/2.0, self.estado_ia], dtype=np.float32)

    def step(self, action_ia, action_rival=None, ia_move_name=None):
        # Aseguramos que las acciones sean enteros simples
        if isinstance(action_ia, (np.ndarray, np.generic)):
            action_ia = int(action_ia.item())
        
        if action_rival is None:
            # Bot rival simple
            action_rival = self.tipo_rival_real if random.random() > 0.2 else random.randint(0, 3)
        else:
            if isinstance(action_rival, (np.ndarray, np.generic)):
                action_rival = int(action_rival.item())

        ia_move_log = ia_move_name if ia_move_name else "Ataque IA"
        rival_move_log = "Ataque Rival"

        # Lógica de Daño IA -> Rival
        mult = self.tabla_tipos.get(self.tipo_ia, {}).get(self.tipo_rival_real, 1.0)
        self.hp_rival -= (0.15 * mult)
        self.tipo_rival_visible = float(self.tipo_rival_real)

        # Lógica de Daño Rival -> IA
        mult_r = self.tabla_tipos.get(self.tipo_rival_real, {}).get(self.tipo_ia, 1.0)
        self.hp_ia -= (0.15 * mult_r)

        # Registro de efectos
        if action_ia == 3: ia_move_log = "Estado ✨"
        if action_rival == 3: rival_move_log = "Estado ✨"

        self.hp_ia = float(np.clip(self.hp_ia, 0, 1))
        self.hp_rival = float(np.clip(self.hp_rival, 0, 1))
        
        terminated = self.hp_ia <= 0 or self.hp_rival <= 0
        info = {'ia_move': ia_move_log, 'rival_move': rival_move_log}
        
        return self._get_obs(), 0.0, terminated, False, info