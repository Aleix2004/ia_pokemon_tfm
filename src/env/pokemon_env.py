import copy
import random
import sqlite3

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from src.battle_utils import (
        STAT_NAME_MAP,
        TYPE_ORDER,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_index,
        get_type_multiplier,
    )
except ImportError:
    from battle_utils import (
        STAT_NAME_MAP,
        TYPE_ORDER,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_index,
        get_type_multiplier,
    )


ENV_VERSION = "pokemon_env_v1_obs28_act4"
OBSERVATION_SHAPE = (28,)
ACTION_SIZE = 4


def build_move(name, move_type, power, damage_class, target="selected-pokemon", stat_changes=None):
    return {
        "name": name,
        "type": move_type,
        "power": power,
        "accuracy": 100,
        "pp": 20,
        "damage_class": damage_class,
        "target": target,
        "stat_changes": stat_changes or [],
    }


TRAINING_ROSTER = [
    {
        "name": "Charizard",
        "types": ["fire", "flying"],
        "base_stats": {"hp": 78, "atk": 84, "def": 78, "sp_atk": 109, "sp_def": 85, "spd": 100},
        "moves": [
            build_move("Flamethrower", "fire", 90, "special"),
            build_move("Air Slash", "flying", 75, "special"),
            build_move("Dragon Claw", "dragon", 80, "physical"),
            build_move("Will-O-Wisp Lite", "fire", 0, "status", stat_changes=[{"name": "attack", "change": -1}]),
        ],
    },
    {
        "name": "Blastoise",
        "types": ["water"],
        "base_stats": {"hp": 79, "atk": 83, "def": 100, "sp_atk": 85, "sp_def": 105, "spd": 78},
        "moves": [
            build_move("Surf", "water", 90, "special"),
            build_move("Ice Beam", "ice", 90, "special"),
            build_move("Bite", "dark", 60, "physical"),
            build_move("Withdraw", "water", 0, "status", target="user", stat_changes=[{"name": "defense", "change": 1}]),
        ],
    },
    {
        "name": "Venusaur",
        "types": ["grass", "poison"],
        "base_stats": {"hp": 80, "atk": 82, "def": 83, "sp_atk": 100, "sp_def": 100, "spd": 80},
        "moves": [
            build_move("Energy Ball", "grass", 90, "special"),
            build_move("Sludge Bomb", "poison", 90, "special"),
            build_move("Earthquake", "ground", 100, "physical"),
            build_move("Growth", "normal", 0, "status", target="user", stat_changes=[{"name": "attack", "change": 1}, {"name": "special-attack", "change": 1}]),
        ],
    },
    {
        "name": "Pikachu",
        "types": ["electric"],
        "base_stats": {"hp": 35, "atk": 55, "def": 40, "sp_atk": 50, "sp_def": 50, "spd": 90},
        "moves": [
            build_move("Thunderbolt", "electric", 90, "special"),
            build_move("Quick Attack", "normal", 40, "physical"),
            build_move("Iron Tail", "steel", 100, "physical"),
            build_move("Nasty Plot", "dark", 0, "status", target="user", stat_changes=[{"name": "special-attack", "change": 2}]),
        ],
    },
    {
        "name": "Garchomp",
        "types": ["dragon", "ground"],
        "base_stats": {"hp": 108, "atk": 130, "def": 95, "sp_atk": 80, "sp_def": 85, "spd": 102},
        "moves": [
            build_move("Earthquake", "ground", 100, "physical"),
            build_move("Dragon Claw", "dragon", 80, "physical"),
            build_move("Stone Edge", "rock", 100, "physical"),
            build_move("Swords Dance", "normal", 0, "status", target="user", stat_changes=[{"name": "attack", "change": 2}]),
        ],
    },
    {
        "name": "Alakazam",
        "types": ["psychic"],
        "base_stats": {"hp": 55, "atk": 50, "def": 45, "sp_atk": 135, "sp_def": 95, "spd": 120},
        "moves": [
            build_move("Psychic", "psychic", 90, "special"),
            build_move("Shadow Ball", "ghost", 80, "special"),
            build_move("Dazzling Gleam", "fairy", 80, "special"),
            build_move("Calm Mind", "psychic", 0, "status", target="user", stat_changes=[{"name": "special-attack", "change": 1}, {"name": "special-defense", "change": 1}]),
        ],
    },
]


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_turns=40):
        super().__init__()
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=OBSERVATION_SHAPE, dtype=np.float32)

        self.live_battle = False
        self.ia_pokemon = None
        self.rival_pokemon = None
        self.opponent_mode = "random"
        self.opponent_model = None
        self.random_baseline_chance = 0.5
        self.reset()

    def configure_battle(self, ia_pokemon, rival_pokemon):
        self.live_battle = True
        self.ia_pokemon = ia_pokemon
        self.rival_pokemon = rival_pokemon
        self._reset_episode_trackers()
        self._sync_pokemon_state(self.ia_pokemon)
        self._sync_pokemon_state(self.rival_pokemon)

    def clear_battle(self):
        self.live_battle = False
        self.ia_pokemon = None
        self.rival_pokemon = None

    def set_opponent(self, mode="random", model=None, random_baseline_chance=0.5):
        self.opponent_mode = mode
        self.opponent_model = model
        self.random_baseline_chance = float(np.clip(random_baseline_chance, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._reset_episode_trackers()

        if self.live_battle and self.ia_pokemon and self.rival_pokemon:
            self._sync_pokemon_state(self.ia_pokemon)
            self._sync_pokemon_state(self.rival_pokemon)
            return self._get_obs(), {}

        ia_template, rival_template = random.sample(TRAINING_ROSTER, 2)
        self.ia_pokemon = self._build_training_pokemon(ia_template)
        self.rival_pokemon = self._build_training_pokemon(rival_template)
        self._sync_pokemon_state(self.ia_pokemon)
        self._sync_pokemon_state(self.rival_pokemon)
        return self._get_obs(), {}

    def _reset_episode_trackers(self):
        self.turn_count = 0
        self.episode_reward = 0.0
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.episode_kos_for = 0
        self.episode_kos_against = 0
        self.episode_stalled_turns = 0
        self.last_reward_breakdown = {}
        self.estado_rival = 0.0
        self.estado_ia = 0.0
        self.hp_ia = 1.0
        self.hp_rival = 1.0

    def _build_training_pokemon(self, template):
        pokemon = copy.deepcopy(template)
        pokemon["stats"] = dict(pokemon["base_stats"])
        pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
        pokemon["current_hp"] = 1.0
        pokemon["status"] = None
        pokemon["item"] = None
        pokemon["debilitado"] = False
        return pokemon

    def _stage_norm(self, pokemon, stat_name):
        return (pokemon.get("stat_stages", {}).get(stat_name, 0) + 6) / 12.0

    def _status_value(self, pokemon):
        return 0.0 if not pokemon.get("status") else 1.0

    def _stat_norm(self, pokemon, stat_name):
        return min(1.0, pokemon["stats"].get(stat_name, 0) / 255.0)

    def _type_pair(self, pokemon):
        types = pokemon.get("types", []) or ["normal"]
        first = get_type_index(types[0]) / max(1, len(TYPE_ORDER) - 1)
        second = get_type_index(types[1]) / max(1, len(TYPE_ORDER) - 1) if len(types) > 1 else 0.0
        return first, second

    def _get_obs(self, for_rival=False):
        me = self.rival_pokemon if for_rival else self.ia_pokemon
        foe = self.ia_pokemon if for_rival else self.rival_pokemon
        me_t1, me_t2 = self._type_pair(me)
        foe_t1, foe_t2 = self._type_pair(foe)
        obs = np.array(
            [
                float(me.get("current_hp", 1.0)),
                float(foe.get("current_hp", 1.0)),
                me_t1,
                me_t2,
                foe_t1,
                foe_t2,
                self._status_value(me),
                self._status_value(foe),
                self._stage_norm(me, "atk"),
                self._stage_norm(me, "def"),
                self._stage_norm(me, "sp_atk"),
                self._stage_norm(me, "sp_def"),
                self._stage_norm(me, "spd"),
                self._stage_norm(foe, "atk"),
                self._stage_norm(foe, "def"),
                self._stage_norm(foe, "sp_atk"),
                self._stage_norm(foe, "sp_def"),
                self._stage_norm(foe, "spd"),
                self._stat_norm(me, "atk"),
                self._stat_norm(me, "def"),
                self._stat_norm(me, "sp_atk"),
                self._stat_norm(me, "sp_def"),
                self._stat_norm(me, "spd"),
                self._stat_norm(foe, "atk"),
                self._stat_norm(foe, "def"),
                self._stat_norm(foe, "sp_atk"),
                self._stat_norm(foe, "sp_def"),
                self._stat_norm(foe, "spd"),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action_ia, action_rival=None, ia_move_name=None):
        action_ia = self._normalize_action(action_ia, self.ia_pokemon)
        if action_rival is None:
            action_rival = self._select_opponent_action()
        else:
            action_rival = self._normalize_action(action_rival, self.rival_pokemon)
        return self._run_turn(action_ia, action_rival)

    def _select_opponent_action(self):
        if self.live_battle:
            move_total = max(1, len(self.rival_pokemon.get("moves", [])))
            return random.randint(0, min(3, move_total - 1))

        if self.opponent_mode == "model":
            if self.opponent_model is None:
                raise RuntimeError("Opponent mode 'model' requires a loaded PPO model")
            action, _ = self.opponent_model.predict(self._get_obs(for_rival=True), deterministic=False)
            return self._normalize_action(action, self.rival_pokemon)

        if self.opponent_mode == "mixed":
            if self.opponent_model is None:
                raise RuntimeError("Opponent mode 'mixed' requires a loaded PPO model")
            if random.random() > self.random_baseline_chance:
                action, _ = self.opponent_model.predict(self._get_obs(for_rival=True), deterministic=False)
                return self._normalize_action(action, self.rival_pokemon)
            return self._select_greedy_action()

        if self.opponent_mode == "greedy":
            return self._select_greedy_action()

        if self.opponent_mode == "random":
            move_total = max(1, len(self.rival_pokemon.get("moves", [])))
            return random.randint(0, min(3, move_total - 1))

        raise RuntimeError(f"Unsupported opponent mode: {self.opponent_mode}")

    def _select_greedy_action(self):
        move_scores = []
        for idx, move in enumerate(self.rival_pokemon.get("moves", [])):
            effectiveness = get_type_multiplier(move.get("type"), self.ia_pokemon.get("types", []))
            power = move.get("power") or 0
            move_scores.append((effectiveness * max(1, power), idx))
        if not move_scores:
            return 0
        move_scores.sort(reverse=True)
        return move_scores[0][1]

    def switch_turn(self, side, new_active_pokemon, opponent_action=None):
        if not (self.live_battle and self.ia_pokemon and self.rival_pokemon):
            raise RuntimeError("switch_turn is only available in live battle mode")

        reward = 0.0
        if side == "rival":
            old_name = self.rival_pokemon["name"]
            self.rival_pokemon = new_active_pokemon
            self._sync_pokemon_state(self.rival_pokemon)
            switch_log = f"{old_name} switched out for {new_active_pokemon['name']}"
            attack_result = None
            old_hp_rival = self.hp_rival
            old_hp_ia = self.hp_ia
            if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
                ia_action = self._normalize_action(opponent_action, self.ia_pokemon)
                attack_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, self.ia_pokemon["moves"][ia_action])
                reward_data = self._compute_reward(old_hp_ia, old_hp_rival, switch_penalty=0.0)
                reward = reward_data["reward"]
            self._write_live_log(
                ia_result=attack_result,
                rival_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            )
            info = self._build_turn_info(
                reward=reward,
                reward_data=self.last_reward_breakdown,
                old_hp_ia=old_hp_ia,
                old_hp_rival=old_hp_rival,
                ia_result=attack_result,
                rival_result={"log": switch_log, "type": "", "effectiveness_label": ""},
                switch_log=switch_log,
            )
            return self._get_obs(), reward, self._is_battle_over(), False, info

        old_name = self.ia_pokemon["name"]
        self.ia_pokemon = new_active_pokemon
        self._sync_pokemon_state(self.ia_pokemon)
        switch_log = f"{old_name} switched out for {new_active_pokemon['name']}"
        old_hp_rival = self.hp_rival
        old_hp_ia = self.hp_ia
        attack_result = None
        if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
            rival_action = self._normalize_action(opponent_action, self.rival_pokemon)
            attack_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, self.rival_pokemon["moves"][rival_action])
        reward_data = self._compute_reward(old_hp_ia, old_hp_rival, switch_penalty=0.05)
        reward = reward_data["reward"]
        self._write_live_log(
            ia_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            rival_result=attack_result,
        )
        info = self._build_turn_info(
            reward=reward,
            reward_data=reward_data,
            old_hp_ia=old_hp_ia,
            old_hp_rival=old_hp_rival,
            ia_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            rival_result=attack_result,
            switch_log=switch_log,
        )
        return self._get_obs(), reward, self._is_battle_over(), False, info

    def _run_turn(self, action_ia, action_rival):
        self.turn_count += 1
        old_hp_ia = float(self.ia_pokemon.get("current_hp", 1.0))
        old_hp_rival = float(self.rival_pokemon.get("current_hp", 1.0))
        ia_move = self.ia_pokemon["moves"][action_ia]
        rival_move = self.rival_pokemon["moves"][action_rival]

        ia_result = None
        rival_result = None
        ia_speed = self.ia_pokemon["stats"].get("spd", 1)
        rival_speed = self.rival_pokemon["stats"].get("spd", 1)
        ia_first = ia_speed >= rival_speed if ia_speed != rival_speed else random.random() < 0.5

        if ia_first:
            ia_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, ia_move)
            if self.rival_pokemon.get("current_hp", 0.0) > 0:
                rival_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, rival_move)
        else:
            rival_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, rival_move)
            if self.ia_pokemon.get("current_hp", 0.0) > 0:
                ia_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, ia_move)

        reward_data = self._compute_reward(old_hp_ia, old_hp_rival)
        reward = reward_data["reward"]

        if self.live_battle:
            self._write_live_log(ia_result=ia_result, rival_result=rival_result)

        terminated = self._is_battle_over()
        truncated = self.turn_count >= self.max_turns
        info = self._build_turn_info(
            reward=reward,
            reward_data=reward_data,
            old_hp_ia=old_hp_ia,
            old_hp_rival=old_hp_rival,
            ia_result=ia_result,
            rival_result=rival_result,
        )
        if truncated and not terminated:
            info["is_win"] = self.hp_rival < self.hp_ia
        return self._get_obs(), reward, terminated, truncated, info

    def _is_battle_over(self):
        return self.hp_ia <= 0 or self.hp_rival <= 0

    def _compute_reward(self, old_hp_ia, old_hp_rival, switch_penalty=0.0):
        damage_to_rival = max(0.0, old_hp_rival - self.hp_rival)
        damage_to_ia = max(0.0, old_hp_ia - self.hp_ia)
        self.episode_damage_dealt += damage_to_rival
        self.episode_damage_received += damage_to_ia
        if damage_to_rival <= 1e-9 and damage_to_ia <= 1e-9:
            self.episode_stalled_turns += 1

        reward = 0.0
        reward += 0.35 * damage_to_rival
        reward -= 0.35 * damage_to_ia

        ko_bonus = 0.35 if old_hp_rival > 0 and self.hp_rival <= 0 else 0.0
        faint_penalty = 0.35 if old_hp_ia > 0 and self.hp_ia <= 0 else 0.0
        if ko_bonus > 0:
            self.episode_kos_for += 1
        if faint_penalty > 0:
            self.episode_kos_against += 1
        reward += ko_bonus
        reward -= faint_penalty

        reward -= 0.01
        reward -= switch_penalty

        terminal_bonus = 0.0
        if self.hp_rival <= 0:
            terminal_bonus = 1.0
            reward += terminal_bonus
        elif self.hp_ia <= 0:
            terminal_bonus = -1.0
            reward += terminal_bonus

        self.episode_reward += reward
        self.last_reward_breakdown = {
            "reward": reward,
            "damage_dealt_reward": 0.35 * damage_to_rival,
            "damage_taken_penalty": -0.35 * damage_to_ia,
            "ko_bonus": ko_bonus,
            "faint_penalty": -faint_penalty,
            "switch_penalty": -switch_penalty,
            "stall_penalty": -0.01,
            "terminal_bonus": terminal_bonus,
        }
        return self.last_reward_breakdown

    def _build_turn_info(self, reward, reward_data, old_hp_ia, old_hp_rival, ia_result, rival_result, switch_log=""):
        terminated = self._is_battle_over()
        info = {
            "ia_move": ia_result["log"] if ia_result else "Skipped",
            "rival_move": rival_result["log"] if rival_result else "Skipped",
            "ia_move_type": ia_result["type"] if ia_result else "",
            "rival_move_type": rival_result["type"] if rival_result else "",
            "ia_effectiveness": ia_result["effectiveness_label"] if ia_result else "",
            "rival_effectiveness": rival_result["effectiveness_label"] if rival_result else "",
            "hp_change_ia": old_hp_ia - self.hp_ia,
            "hp_change_rival": old_hp_rival - self.hp_rival,
            "turn": self.turn_count,
            "reward": reward,
            "reward_breakdown": reward_data,
            "damage_dealt": self.episode_damage_dealt,
            "damage_received": self.episode_damage_received,
            "ko_count": self.episode_kos_for,
            "ko_received": self.episode_kos_against,
            "stalled_turns": self.episode_stalled_turns,
            "switch_log": switch_log,
            "is_win": terminated and self.hp_rival <= 0,
        }
        if terminated:
            info.update(
                {
                    "episode_reward": self.episode_reward,
                    "episode_length": self.turn_count,
                    "final_hp_ia": self.hp_ia,
                    "final_hp_rival": self.hp_rival,
                    "ko_count": self.episode_kos_for,
                    "ko_received": self.episode_kos_against,
                    "stalled_turns": self.episode_stalled_turns,
                }
            )
        return info

    def _write_live_log(self, ia_result, rival_result):
        try:
            conn = sqlite3.connect("pokemon_bigdata.db")
            curr = conn.cursor()
            curr.execute(
                """
                INSERT INTO v_logs (
                    ia_move_name, rival_move, ia_move_type, rival_move_type,
                    ia_effectiveness, rival_effectiveness, hp_ia, hp_rival, reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ia_result["log"] if ia_result else "Skipped",
                    rival_result["log"] if rival_result else "Skipped",
                    ia_result["type"] if ia_result else "",
                    rival_result["type"] if rival_result else "",
                    ia_result["effectiveness_label"] if ia_result else "",
                    rival_result["effectiveness_label"] if rival_result else "",
                    self.hp_ia,
                    self.hp_rival,
                    self.last_reward_breakdown.get("reward", 0.0),
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _normalize_action(self, action, pokemon):
        if isinstance(action, (np.ndarray, np.generic)):
            action = int(action.item())
        move_count = max(1, len(pokemon.get("moves", [])))
        return max(0, min(int(action), move_count - 1))

    def _execute_move(self, attacker, defender, move):
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
        damage_ratio = 0.0 if effectiveness == 0 else min(0.95, max(0.01, damage / max(1, defender["base_stats"]["hp"] * 2.5)))
        defender["current_hp"] = float(np.clip(defender.get("current_hp", 1.0) - damage_ratio, 0.0, 1.0))
        defender["debilitado"] = defender["current_hp"] <= 0
        self._sync_pokemon_state(attacker)
        self._sync_pokemon_state(defender)
        log = f"{attacker['name']} used {move['name']} ({format_name(move_type)}) [{describe_effectiveness(effectiveness)}]"
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
        pokemon["current_hp"] = float(np.clip(pokemon.get("current_hp", 1.0), 0.0, 1.0))
        pokemon["debilitado"] = pokemon["current_hp"] <= 0
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
