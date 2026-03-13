import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        # Acciones: 0: Fuego, 1: Agua, 2: Planta, 3: ONDA TRUENO (Parálisis)
        self.action_space = spaces.Discrete(4)
        
        # Obs: [Vida IA, Vida Rival, Tipo IA, Tipo Rival Visible, Estado Rival]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.tabla_tipos = {
            0: {2: 2.0, 1: 0.5}, # Fuego
            1: {0: 2.0, 2: 0.5}, # Agua
            2: {1: 2.0, 0: 0.5}, # Planta
            3: {1: 2.0}          # Eléctrico (Rival)
        }
        self.nombres_tipos = {0: "Fuego", 1: "Agua", 2: "Planta", 3: "Electrico"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tipo_ia = random.randint(0, 2)
        self.tipo_rival_real = random.randint(0, 3)
        self.tipo_rival_visible = -1.0 
        self.hp_ia = 1.0
        self.hp_rival = 1.0
        self.estado_rival = 0.0 # Parálisis del rival
        self.estado_ia = 0.0    # Parálisis de la IA
        return self._get_obs(), {}

    def _get_obs(self, for_rival=False):
        """Devuelve la observación. Si for_rival es True, invierte la perspectiva."""
        if not for_rival:
            visible = self.tipo_rival_visible / 3.0 if self.tipo_rival_visible != -1.0 else 0.5
            return np.array([
                self.hp_ia, 
                self.hp_rival, 
                self.tipo_ia / 2.0, 
                visible, 
                self.estado_rival
            ], dtype=np.float32)
        else:
            # Perspectiva invertida para el contrincante en Self-Play
            return np.array([
                self.hp_rival, 
                self.hp_ia, 
                self.tipo_rival_real / 3.0, 
                self.tipo_ia / 2.0, 
                self.estado_ia
            ], dtype=np.float32)

    def step(self, action_ia, action_rival=None):
        # Asegurar que las acciones sean enteros puros para evitar errores de diccionario
        if isinstance(action_ia, (np.ndarray, np.generic)): 
            action_ia = action_ia.item()
        action_ia = int(action_ia)
        
        if action_rival is not None:
            if isinstance(action_rival, (np.ndarray, np.generic)): 
                action_rival = action_rival.item()
            action_rival = int(action_rival)
        else:
            # Lógica por defecto si no hay acción externa (Self-Play)
            action_rival = self.tipo_rival_real if random.random() > 0.2 else random.randint(0, 3)

        recompensa_ia = 0
        
        # --- LÓGICA DE INTENCIONES ---
        ia_usa_estado = (action_ia == 3)
        ia_cambia = (action_ia != self.tipo_ia and action_ia != 3)
        
        rival_usa_estado = (action_rival == 3)
        rival_cambia = (action_rival != self.tipo_rival_real and action_rival != 3)

        # 1. Fase de Cambios
        if ia_cambia: 
            self.tipo_ia = action_ia
            self.hp_ia -= 0.02
            recompensa_ia -= 0.1
        
        if rival_cambia: 
            self.tipo_rival_real = action_rival
            self.tipo_rival_visible = -1.0

        # 2. Fase de Estados (Onda Trueno)
        if ia_usa_estado:
            if self.estado_rival == 0:
                self.estado_rival = 1.0
                recompensa_ia += 1.0
            else:
                recompensa_ia -= 0.5
        
        if rival_usa_estado and self.estado_ia == 0:
            self.estado_ia = 1.0

        # 3. Probabilidad de Parálisis
        ia_puede_atacar = not (self.estado_ia == 1.0 and random.random() < 0.25)
        rival_puede_atacar = not (self.estado_rival == 1.0 and random.random() < 0.25)

        # 4. Fase de Ataque
        if not ia_cambia and not ia_usa_estado and ia_puede_atacar:
            mult = self.tabla_tipos.get(self.tipo_ia, {}).get(self.tipo_rival_real, 1.0)
            daño = 0.15 * mult
            self.hp_rival -= daño
            recompensa_ia += daño * 2
            if mult > 1.0: recompensa_ia += 0.5
            self.tipo_rival_visible = float(self.tipo_rival_real)

        if not rival_cambia and not rival_usa_estado and rival_puede_atacar:
            mult_rival = self.tabla_tipos.get(self.tipo_rival_real, {}).get(self.tipo_ia, 1.0)
            daño_recibido = 0.15 * mult_rival
            self.hp_ia -= daño_recibido
            recompensa_ia -= daño_recibido * 2
            if mult_rival > 1.0 and not ia_cambia:
                recompensa_ia -= 0.75

        # --- FINAL DEL TURNO ---
        self.hp_ia = float(np.clip(self.hp_ia, 0, 1))
        self.hp_rival = float(np.clip(self.hp_rival, 0, 1))
        
        terminated = self.hp_ia <= 0 or self.hp_rival <= 0
        if terminated:
            victoria = self.hp_rival <= 0
            recompensa_ia += 10 if victoria else -10

        return self._get_obs(), recompensa_ia, terminated, False, {
            'is_win': self.hp_rival <= 0,
            'rival_type': self.nombres_tipos.get(self.tipo_rival_real, "Desconocido")
        }