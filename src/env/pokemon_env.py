import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        # 4 acciones: Ataque 1, 2, 3 o 4
        self.action_space = spaces.Discrete(4)
        # Observación: [Vida propia, Vida rival, Ventaja, Velocidad]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Empezamos con vida al 100% (1.0)
        self.state = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # Lógica simplificada: restamos vida al rival
        self.state[1] -= 0.2 
        # El rival nos resta vida a nosotros
        self.state[0] -= 0.1 
        
        # Recompensas
        reward = 0.1
        terminated = False
        if self.state[1] <= 0: # Ganamos
            reward = 10.0
            terminated = True
        elif self.state[0] <= 0: # Perdemos
            reward = -5.0
            terminated = True
            
        return self.state, reward, terminated, False, {}