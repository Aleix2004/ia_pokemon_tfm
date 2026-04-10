import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sqlite3
try:
    from src.battle_utils import (
        STAT_NAME_MAP,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_index,
        get_type_multiplier,
    )
except ImportError:
    from battle_utils import (
        STAT_NAME_MAP,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_index,
        get_type_multiplier,
    )

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.tabla_tipos = {
            0: {2: 2.0, 1: 0.5}, # Fuego
            1: {0: 2.0, 2: 0.5}, # Agua
            2: {1: 2.0, 0: 0.5}, # Planta
            3: {1: 2.0}          # Eléctrico
        }
        self.nombres_tipos = {0: "Fuego", 1: "Agua", 2: "Planta", 3: "Electrico"}
        self.live_battle = False
        self.ia_pokemon = None
        self.rival_pokemon = None
        self.reset()

    def configure_battle(self, ia_pokemon, rival_pokemon):
        self.live_battle = True
        self.ia_pokemon = ia_pokemon
        self.rival_pokemon = rival_pokemon
        self.hp_ia = float(ia_pokemon.get("current_hp", 1.0))
        self.hp_rival = float(rival_pokemon.get("current_hp", 1.0))

    def clear_battle(self):
        self.live_battle = False
        self.ia_pokemon = None
        self.rival_pokemon = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.live_battle and self.ia_pokemon and self.rival_pokemon:
            self.hp_ia = float(self.ia_pokemon.get("current_hp", 1.0))
            self.hp_rival = float(self.rival_pokemon.get("current_hp", 1.0))
            self.estado_rival = 0.0
            self.estado_ia = 0.0
            return self._get_obs(), {}
        self.tipo_ia = random.randint(0, 2)
        self.tipo_rival_real = random.randint(0, 3)
        self.tipo_rival_visible = -1.0 
        self.hp_ia = 1.0
        self.hp_rival = 1.0
        self.estado_rival = 0.0 
        self.estado_ia = 0.0    
        return self._get_obs(), {}

    def _get_obs(self, for_rival=False):
        if self.live_battle and self.ia_pokemon and self.rival_pokemon:
            ia_type = get_type_index((self.ia_pokemon.get("types") or ["normal"])[0])
            rival_type = get_type_index((self.rival_pokemon.get("types") or ["normal"])[0])
            visible = rival_type / 17.0
            if not for_rival:
                return np.array([self.hp_ia, self.hp_rival, ia_type / 17.0, visible, self.estado_rival], dtype=np.float32)
            return np.array([self.hp_rival, self.hp_ia, rival_type / 17.0, ia_type / 17.0, self.estado_ia], dtype=np.float32)
        if not for_rival:
            visible = self.tipo_rival_visible / 3.0 if self.tipo_rival_visible != -1.0 else 0.5
            return np.array([self.hp_ia, self.hp_rival, self.tipo_ia/2.0, visible, self.estado_rival], dtype=np.float32)
        else:
            return np.array([self.hp_rival, self.hp_ia, self.tipo_rival_real/3.0, self.tipo_ia/2.0, self.estado_ia], dtype=np.float32)

    def step(self, action_ia, action_rival=None, ia_move_name=None):
        if self.live_battle and self.ia_pokemon and self.rival_pokemon:
            return self._step_live_battle(action_ia, action_rival)
        if isinstance(action_ia, (np.ndarray, np.generic)):
            action_ia = int(action_ia.item())
        
        if action_rival is None:
            action_rival = self.tipo_rival_real if random.random() > 0.2 else random.randint(0, 3)
        else:
            if isinstance(action_rival, (np.ndarray, np.generic)):
                action_rival = int(action_rival.item())

        ia_move_log = ia_move_name if ia_move_name else f"Ataque {self.nombres_tipos.get(self.tipo_ia, 'IA')}"
        rival_move_log = f"Ataque {self.nombres_tipos.get(action_rival, 'Rival')}"

        # --- LÓGICA DE DAÑO CON VARIABILIDAD ---
        # IA -> Rival
        mult = self.tabla_tipos.get(self.tipo_ia, {}).get(self.tipo_rival_real, 1.0)
        dmg_ia = random.uniform(0.12, 0.18) * mult
        self.hp_rival -= dmg_ia
        self.tipo_rival_visible = float(self.tipo_rival_real)

        # Rival -> IA
        mult_r = self.tabla_tipos.get(self.tipo_rival_real, {}).get(self.tipo_ia, 1.0)
        dmg_riv = random.uniform(0.10, 0.16) * mult_r
        self.hp_ia -= dmg_riv

        self.hp_ia = float(np.clip(self.hp_ia, 0, 1))
        self.hp_rival = float(np.clip(self.hp_rival, 0, 1))
        
        # --- LOG EN SQLITE ---
        try:
            conn = sqlite3.connect('pokemon_bigdata.db')
            curr = conn.cursor()
            curr.execute('INSERT INTO v_logs (ia_move_name, rival_move, hp_ia, hp_rival, reward) VALUES (?, ?, ?, ?, ?)',
                         (ia_move_log, rival_move_log, self.hp_ia, self.hp_rival, 0.0))
            conn.commit()
            conn.close()
        except:
            pass

        terminated = self.hp_ia <= 0 or self.hp_rival <= 0
        info = {'ia_move': ia_move_log, 'rival_move': rival_move_log}
        
        return self._get_obs(), 0.0, terminated, False, info

    def switch_turn(self, side, new_active_pokemon, opponent_action=None):
        if not (self.live_battle and self.ia_pokemon and self.rival_pokemon):
            raise RuntimeError("switch_turn is only available in live battle mode")

        if side == "rival":
            old_name = self.rival_pokemon["name"]
            self.rival_pokemon = new_active_pokemon
            self.hp_rival = float(new_active_pokemon.get("current_hp", 1.0))
            switch_log = f"{old_name} switched out for {new_active_pokemon['name']}"
            attack_result = None
            old_hp_rival = self.hp_rival
            old_hp_ia = self.hp_ia
            if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
                ia_action = self._normalize_action(opponent_action, self.ia_pokemon)
                ia_move = self.ia_pokemon["moves"][ia_action]
                attack_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, ia_move, "ia")
            self._write_live_log(
                ia_result=attack_result,
                rival_result={
                    "log": switch_log,
                    "type": "",
                    "effectiveness_label": "",
                },
            )
            info = {
                "switch_log": switch_log,
                "ia_move": attack_result["log"] if attack_result else "No attack",
                "rival_move": switch_log,
                "ia_move_type": attack_result["type"] if attack_result else "",
                "rival_move_type": "",
                "ia_effectiveness": attack_result["effectiveness_label"] if attack_result else "",
                "rival_effectiveness": "",
                "hp_change_ia": old_hp_ia - self.hp_ia,
                "hp_change_rival": old_hp_rival - self.hp_rival,
            }
            return self._get_obs(), 0.0, self.hp_ia <= 0 or self.hp_rival <= 0, False, info

        old_name = self.ia_pokemon["name"]
        self.ia_pokemon = new_active_pokemon
        self.hp_ia = float(new_active_pokemon.get("current_hp", 1.0))
        switch_log = f"{old_name} switched out for {new_active_pokemon['name']}"
        attack_result = None
        old_hp_rival = self.hp_rival
        old_hp_ia = self.hp_ia
        if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
            rival_action = self._normalize_action(opponent_action, self.rival_pokemon)
            rival_move = self.rival_pokemon["moves"][rival_action]
            attack_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, rival_move, "rival")
        self._write_live_log(
            ia_result={
                "log": switch_log,
                "type": "",
                "effectiveness_label": "",
            },
            rival_result=attack_result,
        )
        info = {
            "switch_log": switch_log,
            "ia_move": switch_log,
            "rival_move": attack_result["log"] if attack_result else "No attack",
            "ia_move_type": "",
            "rival_move_type": attack_result["type"] if attack_result else "",
            "ia_effectiveness": "",
            "rival_effectiveness": attack_result["effectiveness_label"] if attack_result else "",
            "hp_change_ia": old_hp_ia - self.hp_ia,
            "hp_change_rival": old_hp_rival - self.hp_rival,
        }
        return self._get_obs(), 0.0, self.hp_ia <= 0 or self.hp_rival <= 0, False, info

    def _step_live_battle(self, action_ia, action_rival=None):
        action_ia = self._normalize_action(action_ia, self.ia_pokemon)
        if action_rival is None:
            move_total = max(1, len(self.rival_pokemon.get("moves", [])))
            action_rival = random.randint(0, min(3, move_total - 1))
        else:
            action_rival = self._normalize_action(action_rival, self.rival_pokemon)

        ia_move = self.ia_pokemon["moves"][action_ia]
        rival_move = self.rival_pokemon["moves"][action_rival]

        old_hp_ia = self.hp_ia
        old_hp_rival = self.hp_rival

        ia_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, ia_move, "ia")
        rival_result = None
        if self.hp_rival > 0:
            rival_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, rival_move, "rival")

        self.hp_ia = float(self.ia_pokemon.get("current_hp", self.hp_ia))
        self.hp_rival = float(self.rival_pokemon.get("current_hp", self.hp_rival))
        self.estado_rival = 1.0 if ia_result["effectiveness"] > 1 else 0.0
        self.estado_ia = 1.0 if rival_result and rival_result["effectiveness"] > 1 else 0.0

        self._write_live_log(ia_result=ia_result, rival_result=rival_result)

        terminated = self.hp_ia <= 0 or self.hp_rival <= 0
        info = {
            "ia_move": ia_result["log"],
            "rival_move": rival_result["log"] if rival_result else "Skipped",
            "ia_move_type": ia_result["type"],
            "rival_move_type": rival_result["type"] if rival_result else "",
            "ia_effectiveness": ia_result["effectiveness_label"],
            "rival_effectiveness": rival_result["effectiveness_label"] if rival_result else "",
            "hp_change_ia": old_hp_ia - self.hp_ia,
            "hp_change_rival": old_hp_rival - self.hp_rival,
        }
        return self._get_obs(), 0.0, terminated, False, info

    def _write_live_log(self, ia_result, rival_result):
        try:
            conn = sqlite3.connect('pokemon_bigdata.db')
            curr = conn.cursor()
            curr.execute(
                '''
                INSERT INTO v_logs (
                    ia_move_name, rival_move, ia_move_type, rival_move_type,
                    ia_effectiveness, rival_effectiveness, hp_ia, hp_rival, reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    ia_result["log"] if ia_result else "Skipped",
                    rival_result["log"] if rival_result else "Skipped",
                    ia_result["type"] if ia_result else "",
                    rival_result["type"] if rival_result else "",
                    ia_result["effectiveness_label"] if ia_result else "",
                    rival_result["effectiveness_label"] if rival_result else "",
                    self.hp_ia,
                    self.hp_rival,
                    0.0,
                ),
            )
            conn.commit()
            conn.close()
        except:
            pass

    def _normalize_action(self, action, pokemon):
        if isinstance(action, (np.ndarray, np.generic)):
            action = int(action.item())
        move_count = max(1, len(pokemon.get("moves", [])))
        return max(0, min(int(action), move_count - 1))

    def _execute_move(self, attacker, defender, move, side):
        move_type = move.get("type", "normal")
        effectiveness = get_type_multiplier(move_type, defender.get("types", []))
        damage_class = move.get("damage_class", "status")
        power = move.get("power") or 0

        if damage_class == "status" or power == 0:
            self._apply_move_effects(attacker, defender, move)
            log = f"{attacker['name']} used {move['name']} ({format_name(move_type)})"
            if move.get("stat_changes"):
                log += f" [{self._format_stat_changes(move)}]"
            return {
                "log": log,
                "type": format_name(move_type),
                "effectiveness": effectiveness,
                "effectiveness_label": describe_effectiveness(effectiveness),
            }

        attack_key = "atk" if damage_class == "physical" else "sp_atk"
        defense_key = "def" if damage_class == "physical" else "sp_def"
        attack_stat = max(1, attacker["stats"].get(attack_key, 1))
        defense_stat = max(1, defender["stats"].get(defense_key, 1))
        stab = 1.5 if move_type in attacker.get("types", []) else 1.0
        random_factor = random.uniform(0.92, 1.0)
        damage = (((22 * power * attack_stat / defense_stat) / 50) + 2) * stab * effectiveness * random_factor
        if effectiveness == 0:
            damage_ratio = 0.0
        else:
            damage_ratio = min(0.95, max(0.01, damage / max(1, defender["base_stats"]["hp"] * 2.5)))
        defender["current_hp"] = float(np.clip(defender.get("current_hp", 1.0) - damage_ratio, 0, 1))
        self._sync_pokemon_state(attacker)
        self._sync_pokemon_state(defender)
        log = (
            f"{attacker['name']} used {move['name']} ({format_name(move_type)})"
            f" [{describe_effectiveness(effectiveness)}]"
        )
        return {
            "log": log,
            "type": format_name(move_type),
            "effectiveness": effectiveness,
            "effectiveness_label": describe_effectiveness(effectiveness),
        }

    def _apply_move_effects(self, attacker, defender, move):
        changes = move.get("stat_changes", [])
        target = (move.get("target") or "").lower()
        target_pokemon = attacker if target in {"user", "users-field", "entire-field", "ally"} else defender
        for change in changes:
            stat_name = STAT_NAME_MAP.get(change.get("name"))
            if not stat_name:
                continue
            target_pokemon["stat_stages"][stat_name] = max(
                -6,
                min(6, target_pokemon["stat_stages"].get(stat_name, 0) + change.get("change", 0)),
            )
        self._sync_pokemon_state(attacker)
        self._sync_pokemon_state(defender)

    def _sync_pokemon_state(self, pokemon):
        pokemon["stats"] = apply_stat_stages(pokemon["base_stats"], pokemon.get("stat_stages", {}))
        if pokemon is self.ia_pokemon:
            self.hp_ia = float(pokemon.get("current_hp", self.hp_ia))
        elif pokemon is self.rival_pokemon:
            self.hp_rival = float(pokemon.get("current_hp", self.hp_rival))

    def _format_stat_changes(self, move):
        parts = []
        for change in move.get("stat_changes", []):
            stat_name = STAT_NAME_MAP.get(change.get("name"), change.get("name", "stat"))
            delta = change.get("change", 0)
            if delta:
                parts.append(f"{stat_name} {delta:+d}")
        return ", ".join(parts)
