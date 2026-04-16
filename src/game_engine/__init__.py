"""
src/game_engine
~~~~~~~~~~~~~~~
Game Logic Layer — Pokémon Showdown-style battle engine.

This package is STRICTLY SEPARATE from the RL training environment
(src/env/pokemon_env.py).  It exists ONLY to power the Streamlit
dashboard and optional offline simulations.

Dependency rules:
  - game_engine  MAY import from  src/battle_utils.py
  - game_engine  MAY import from  src/battle_mechanics.py
  - game_engine  MUST NOT import from  src/env/
  - src/env/     MUST NOT import from  src/game_engine/

The PPO models remain fully compatible: BattleEngine._get_obs()
produces the same 28-dim vector as PokemonEnv._get_obs() via
the shared obs_builder module.
"""

from src.game_engine.battle_engine import BattleEngine
from src.game_engine.obs_builder import build_obs_28

__all__ = ["BattleEngine", "build_obs_28"]
