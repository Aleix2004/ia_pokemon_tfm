import copy
import random
import sqlite3
import warnings

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
            # Flash Cannon replaces Bite: Steel/Special (80 pwr) covers Fairy/Ice,
            # consistent with Blastoise's Sp.Atk stat — no longer wastes a slot on
            # a low-power physical move that ignores the right damage class.
            build_move("Flash Cannon", "steel", 80, "special"),
            # Toxic replaces Withdraw: Blastoise is a special tank; threatening
            # Toxic every turn is far more competitive than a +1 Def self-boost.
            build_move("Toxic", "poison", 0, "status", stat_changes=[{"name": "special-defense", "change": 0}]),
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
            # Grass Knot replaces Iron Tail: Iron Tail is physical (wrong class for
            # Nasty Plot Pikachu) with only 75 % accuracy.  Grass Knot is Special
            # and covers Water / Ground / Rock — the three most common types that
            # resist Electric.  Variable power is modelled here as 60 (conservative
            # typical value; real Grass Knot can be up to 120 vs heavy targets).
            build_move("Grass Knot", "grass", 60, "special"),
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
        """Set the active Pokémon for the current live-battle turn.

        Called by the dashboard before every combat step and after a switch.
        Only updates the active Pokémon references and syncs their derived
        stats / HP — it does NOT reset episode trackers.  Episode trackers
        are reset once per full battle in __init__ → reset(), so repeated
        calls to configure_battle mid-battle do not corrupt turn_count,
        episode_damage_dealt, or other cumulative statistics.
        """
        self.live_battle = True
        self.ia_pokemon = ia_pokemon
        self.rival_pokemon = rival_pokemon
        # Sync derived stats (apply_stat_stages) and clamp current_hp.
        # _sync_pokemon_state also keeps self.hp_ia / self.hp_rival in sync
        # with the pokemon dicts, so HP is preserved across turns.
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
        # ── Advanced reward tracking ─────────────────────────────────────────
        # Exponential moving average of (damage_dealt − damage_taken).
        # Captures momentum/tempo: positive = IA is pressuring, negative = IA is pressured.
        self.damage_momentum = 0.0
        # Counts voluntary IA switches this episode for anti-spam fatigue.
        self.switches_this_episode = 0
        # Consecutive turns with meaningful HP advantage (hp_ia > hp_rival + 0.05).
        # Resets when the IA loses its lead.  Used by the sustained-dominance bonus
        # to reward long strategic control of the fight rather than burst KOs.
        self.consecutive_advantage = 0

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

    def _stat_norm(self, pokemon, stat_name):
        return min(1.0, pokemon["stats"].get(stat_name, 0) / 255.0)

    def _type_pair(self, pokemon):
        types = pokemon.get("types", []) or ["normal"]
        first = get_type_index(types[0]) / max(1, len(TYPE_ORDER) - 1)
        second = get_type_index(types[1]) / max(1, len(TYPE_ORDER) - 1) if len(types) > 1 else 0.0
        return first, second

    def _matchup_norm(self, me, foe):
        """
        Best type effectiveness of me's damaging moves vs foe, normalised to [0, 1].

        0.000 = immune (0×)  |  0.250 = neutral (1×)  |  0.500 = 2× SE  |  1.000 = 4× SE

        This is the shared switching signal: the PPO can learn that a value of 0.25
        (neutral) vs an opponent value of 0.5 (2× SE against it) means the current
        matchup is losing and switching is desirable.
        """
        max_eff = max(
            (
                get_type_multiplier(m.get("type", "normal"), foe.get("types", []))
                for m in me.get("moves", [])
                if (m.get("power") or 0) > 0
            ),
            default=1.0,
        )
        return float(np.clip(max_eff / 4.0, 0.0, 1.0))

    def _compute_matchup_score(self, me, foe):
        """
        Type-matchup quality from me's perspective against foe.

        Computes the signed, normalised difference between the best type
        effectiveness me can deal and the best the foe can deal back.

        Returns a float in [-1.0, +1.0]:
            +1.0  me has 4× SE moves AND foe has 0× (immune)
             0.0  perfectly neutral (1× vs 1×)
            -1.0  foe has 4× SE moves AND me has 0× (immune)

        Used by:
          • _compute_reward()   — matchup_shaping & bad_stay_penalty
          • switch_turn()       — smart_switch_quality delta
        """
        ia_best = max(
            (
                get_type_multiplier(m.get("type", "normal"), foe.get("types", []))
                for m in me.get("moves", [])
                if (m.get("power") or 0) > 0
            ),
            default=1.0,
        )
        foe_best = max(
            (
                get_type_multiplier(m.get("type", "normal"), me.get("types", []))
                for m in foe.get("moves", [])
                if (m.get("power") or 0) > 0
            ),
            default=1.0,
        )
        return float(np.clip((ia_best - foe_best) / 4.0, -1.0, 1.0))

    def _estimate_threat_level(self, attacker, defender):
        """
        Heuristic estimate: how many turns until attacker KOs defender?
        Converted to a danger score in [0, 1].

        Returns
        -------
        float in [0, 1]
            0.0  = attacker poses no meaningful threat this turn
            1.0  = attacker is expected to KO defender on the next hit

        Design notes
        ------------
        • Uses the same damage formula as _execute_move with a median random
          factor (0.96) so estimates are slightly pessimistic — the PPO learns
          to be cautious rather than reckless.
        • stat stages are already applied in attacker["stats"] via
          _sync_pokemon_state, so boosts/drops are correctly reflected.
        • Pure computation: no side effects, no state modification.
        • NOT used for decision-making — only as a gradient signal.
        """
        if attacker is None or defender is None:
            return 0.0
        current_hp = float(defender.get("current_hp", 1.0))
        if current_hp <= 0.0:
            return 1.0

        best_dmg_ratio = 0.0
        for move in attacker.get("moves", []):
            power = move.get("power") or 0
            if power == 0:
                continue
            move_type    = move.get("type", "normal")
            effectiveness = get_type_multiplier(move_type, defender.get("types", []))
            if effectiveness == 0:
                continue
            damage_class = move.get("damage_class", "physical")
            attack_key   = "atk" if damage_class == "physical" else "sp_atk"
            defense_key  = "def" if damage_class == "physical" else "sp_def"
            atk = max(1, attacker["stats"].get(attack_key, 1))
            dfn = max(1, defender["stats"].get(defense_key, 1))
            stab = 1.5 if move_type in attacker.get("types", []) else 1.0
            # Median damage estimate (random_factor ≈ 0.96 = centre of [0.92, 1.0])
            dmg   = (((22 * power * atk / dfn) / 50) + 2) * stab * effectiveness * 0.96
            ratio = float(np.clip(dmg / max(1, defender["base_stats"]["hp"] * 2.5), 0.0, 0.95))
            best_dmg_ratio = max(best_dmg_ratio, ratio)

        if best_dmg_ratio <= 0.0:
            return 0.0

        # turns_to_ko: 1.0 = KO this turn, 2.0 = KO in 2 turns, etc.
        turns_to_ko = current_hp / best_dmg_ratio
        # Threat curve: 1.0 when turns_to_ko=1, 0.0 when turns_to_ko≥2.5
        # Gives a smooth anticipation signal — not a hard threshold.
        threat = float(np.clip(1.0 - (turns_to_ko - 1.0) / 1.5, 0.0, 1.0))
        return threat

    def _get_obs(self, for_rival=False):
        """
        Build the 28-dim PPO observation.

        Observation layout (MUST match src/game_engine/obs_builder.build_obs_28):
          [0]    hp_me
          [1]    hp_foe
          [2-3]  type pair me  (normalised indices)
          [4-5]  type pair foe
          [6]    matchup_me  — best move effectiveness me→foe / 4  (switching signal)
          [7]    matchup_foe — best move effectiveness foe→me / 4  (switching signal)
          [8-12] stat stages me   (atk/def/sp_atk/sp_def/spd)
          [13-17] stat stages foe
          [18-22] base stats me  (/255)
          [23-27] base stats foe
        """
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
                self._matchup_norm(me, foe),   # slot 6: how well can ME  hit FOE?
                self._matchup_norm(foe, me),   # slot 7: how well can FOE hit ME?
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
                ia_action     = self._normalize_action(opponent_action, self.ia_pokemon)
                ia_move_used  = self.ia_pokemon["moves"][ia_action]
                attack_result = self._execute_move(self.ia_pokemon, self.rival_pokemon, ia_move_used)
                # Pass IA's move quality so the reward signal is present even on rival-switch turns
                reward_data = self._compute_reward(
                    old_hp_ia, old_hp_rival,
                    switch_penalty=0.0,
                    ia_effectiveness=attack_result.get("effectiveness"),
                    ia_power=ia_move_used.get("power") or 0,
                )
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

        # ── IA switch ──────────────────────────────────────────────────────────
        # Capture pre-switch matchup AND threat scores BEFORE overwriting
        # self.ia_pokemon.  The combined delta rewards improving the matchup
        # AND escaping an imminent KO threat, discouraging both bad matchups
        # and staying in when the opponent is about to score a KO.
        old_ia_pokemon = self.ia_pokemon
        old_matchup    = self._compute_matchup_score(old_ia_pokemon, self.rival_pokemon)
        old_threat     = self._estimate_threat_level(self.rival_pokemon, old_ia_pokemon)

        old_name = old_ia_pokemon["name"]
        self.ia_pokemon = new_active_pokemon
        self._sync_pokemon_state(self.ia_pokemon)

        new_matchup  = self._compute_matchup_score(new_active_pokemon, self.rival_pokemon)
        new_threat   = self._estimate_threat_level(self.rival_pokemon, new_active_pokemon)

        # switch_quality_delta combines:
        #   • matchup delta   — reward switching into a better type matchup
        #   • 0.5 × threat reduction — reward switching away from a KO threat
        # Weighted so threat reduction is worth half a full matchup swing.
        switch_quality_delta = (new_matchup - old_matchup) + 0.5 * (old_threat - new_threat)

        # Track voluntary IA switches for anti-spam fatigue in _compute_reward.
        self.switches_this_episode += 1

        switch_log = f"{old_name} switched out for {new_active_pokemon['name']}"
        old_hp_rival = self.hp_rival
        old_hp_ia = self.hp_ia
        attack_result = None
        if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
            rival_action  = self._normalize_action(opponent_action, self.rival_pokemon)
            attack_result = self._execute_move(self.rival_pokemon, self.ia_pokemon, self.rival_pokemon["moves"][rival_action])
        reward_data = self._compute_reward(
            old_hp_ia, old_hp_rival,
            switch_penalty=0.03,             # reduced from 0.05: smart_switch handles quality
            switch_quality_delta=switch_quality_delta,
        )
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

        # Extract IA move quality context for the move_quality reward component.
        # ia_result is None when the IA was KO'd before attacking (second-mover case).
        ia_eff  = ia_result.get("effectiveness") if ia_result else None
        ia_pwr  = ia_move.get("power") or 0
        ia_acc  = ia_move.get("accuracy") or 100
        reward_data = self._compute_reward(
            old_hp_ia, old_hp_rival,
            ia_effectiveness=ia_eff,
            ia_power=ia_pwr,
            ia_accuracy=ia_acc,
        )
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

    def _compute_reward(
        self,
        old_hp_ia,
        old_hp_rival,
        switch_penalty=0.0,
        ia_effectiveness=None,
        ia_power=0,
        ia_accuracy=100,
        switch_quality_delta=0.0,
    ):
        """
        15-component competitive reward function for strategic PPO training.

        Design philosophy
        -----------------
        The previous reward system produced 2–4 turn "burst KO" episodes because:
          (a) damage_reward (±0.35) + ko_reward (±0.35) made fast bursting
              globally optimal — the total damage signal over the whole battle
              sums to the same value regardless of battle length.
          (b) a flat stall_penalty (−0.01/turn) actively taxed longer battles.

        This version fixes those problems by:
          • Reducing damage/KO weights (less rush incentive).
          • Replacing the flat stall penalty with a conditional one that only
            fires when BOTH sides deal zero damage (true stalling).
          • Adding per-turn survival and HP-lead bonuses that COMPOUND over
            longer battles, making sustained strategic control more rewarding
            than quick bursts.
          • Adding an anti-burst discount on wins that end in ≤ 4 turns.
          • Strengthening threat and matchup signals to encourage proactive
            switching and positional thinking.

        Terminal + KO remain the dominant objectives; all other signals are
        small shaping terms that guide the learning path.

        Component summary (per-step bounds)
        ------------------------------------
         1. damage_reward        ±0.12   — HP differential (reduced to curb burst play)
         2. ko_reward            ±0.22   — KO / faint events (reduced from 0.35)
         3. terminal_bonus       ±1.00   — battle outcome (dominant)
         4. anti_burst_penalty   −0.15   — one-shot discount for wins ≤ 4 turns
         5. stall_penalty        −0.015  — ONLY on true stall turns (both sides deal 0)
         6. survival_bonus       +0.008  — per turn alive (accumulates over long battles)
         7. hp_lead_bonus        +0.015  — proportional to HP advantage (0 when losing)
         8. consecutive_bonus     0..+0.03— compound reward for sustained HP lead
         9. matchup_shaping      ±0.035  — continuous type-advantage gradient
        10. bad_stay_penalty     ≤−0.045 — losing matchup × threat amplifier
        11. temporal_risk        ≤−0.040 — per-turn KO-threat deterrent
        12. momentum_reward      ±0.020  — EMA of damage differential (tempo)
        13. move_quality         −0.06..+0.04 — SE/immunity + accuracy feedback
        14. smart_switch         ±0.06   — quality delta with anti-spam fatigue
        15. switch_cost          −0.03   — base friction on IA switches

        Expected per-episode reward improvement for strategic vs burst play
        -------------------------------------------------------------------
        8-turn strategic win:  ≈ +1.73  (survival + lead + consec. bonuses compound)
        2-turn burst win:      ≈ +1.27  (anti-burst discount + zero accumulated bonuses)
        Delta:                 ≈ +0.46  toward strategic play per episode
        """
        damage_to_rival = max(0.0, old_hp_rival - self.hp_rival)
        damage_to_ia    = max(0.0, old_hp_ia    - self.hp_ia)
        self.episode_damage_dealt    += damage_to_rival
        self.episode_damage_received += damage_to_ia

        # True stall: neither side dealt any damage this turn (both used status
        # moves, or both missed).  The stall_penalty only fires here.
        is_true_stall = damage_to_rival <= 1e-9 and damage_to_ia <= 1e-9
        if is_true_stall:
            self.episode_stalled_turns += 1

        # ── 1. Damage differential ────────────────────────────────────────────
        # Weight reduced from 0.35 → 0.12 so that accumulating damage over
        # multiple turns is no longer vastly better than a single burst.
        # (Total damage dealt ≈ 1 HP regardless of battle length, so the
        # damage_reward sums to ≈0.12 over the whole episode in both cases.
        # The *per-turn* signals below are what now differentiate strategies.)
        damage_reward = 0.12 * damage_to_rival - 0.12 * damage_to_ia

        # ── 2. KO / faint signals ─────────────────────────────────────────────
        # Reduced from 0.35 → 0.22 to dilute the "rush-KO-immediately" gradient.
        # Still a strong signal, but less dominant relative to the cumulative
        # per-turn signals earned over longer strategic battles.
        ko_bonus      = 0.22 if old_hp_rival > 0 and self.hp_rival <= 0 else 0.0
        faint_penalty = 0.22 if old_hp_ia    > 0 and self.hp_ia    <= 0 else 0.0
        if ko_bonus      > 0: self.episode_kos_for     += 1
        if faint_penalty > 0: self.episode_kos_against += 1
        ko_reward = ko_bonus - faint_penalty

        # ── 3. Terminal bonus (primary objective) ─────────────────────────────
        # Unchanged at ±1.0: winning the battle is always the dominant signal.
        terminal_bonus = 0.0
        if   self.hp_rival <= 0: terminal_bonus =  1.0
        elif self.hp_ia    <= 0: terminal_bonus = -1.0

        # ── 4. Anti-burst penalty ─────────────────────────────────────────────
        # Discounts wins that end in ≤ 4 turns.  A fast win is still positive
        # (terminal_bonus +1.0 nets to +0.85 after discount), but the agent
        # learns that strategic multi-turn control is MORE rewarding.
        # Does NOT fire on fast losses (which would perversely reward dying fast).
        anti_burst_penalty = 0.0
        if terminal_bonus > 0 and self.turn_count <= 4:
            anti_burst_penalty = -0.15

        # ── 5. Stall penalty (conditional) ───────────────────────────────────
        # Previously applied EVERY turn (−0.01), which taxed long battles and
        # actively incentivised fast endings.  Now only fires on true stall
        # turns where BOTH sides deal zero damage — status-move spam, misses,
        # or recharge turns where nothing happens.
        stall_penalty = -0.015 if is_true_stall else 0.0

        # ── 6. Survival bonus ─────────────────────────────────────────────────
        # +0.008 for every turn the IA is still alive.  Accumulates over the
        # whole episode so an 8-turn battle earns +0.064 here vs +0.016 for
        # a 2-turn battle — the first of three "compounding length" signals.
        survival_bonus = 0.008 if self.hp_ia > 0 else 0.0

        # ── 7. HP-lead bonus ──────────────────────────────────────────────────
        # Proportional to current HP advantage over the opponent.  Zero when
        # the IA is behind; positive when ahead.  Teaches HP conservation and
        # rewards dominant positioning over the course of the battle.
        #
        #  hp_ia − hp_rival = +1.0  → +0.015/turn  (IA at full, rival at 0)
        #  hp_ia − hp_rival =  0.0  →  0.000/turn  (tied)
        #  hp_ia − hp_rival = −any  →  0.000/turn  (no negative version here;
        #                                            temporal_risk covers threat)
        hp_lead_bonus = 0.015 * max(0.0, self.hp_ia - self.hp_rival)

        # ── 8. Consecutive-advantage bonus ────────────────────────────────────
        # Counts consecutive turns the IA holds a meaningful HP lead (>0.05).
        # The counter grows turn-by-turn (capped at 10) and produces an
        # increasing per-turn bonus.  This is the third compounding signal:
        #
        #   turns at advantage:  0   1   2   3   4   5   6   7   8   9   10
        #   bonus this turn:    .000.003.006.009.012.015.018.021.024.027.030
        #   cumulative (10 t): 0 + 1+2+...+10 = 55 × 0.003 = +0.165 over 10 t
        #   vs 2-turn burst:   ≈ +0.003 (lead barely established by turn 2)
        #
        # The counter DECAYS (−1/turn) when the lead is lost to avoid a
        # persistent unearned bonus after the situation changes.
        if self.hp_ia > self.hp_rival + 0.05:
            self.consecutive_advantage = min(self.consecutive_advantage + 1, 10)
        else:
            self.consecutive_advantage = max(self.consecutive_advantage - 1, 0)
        consecutive_bonus = 0.003 * self.consecutive_advantage   # 0..+0.030/turn

        # ── 9. Matchup shaping (continuous, every turn) ───────────────────────
        # Increased from 0.025 → 0.035 to provide a stronger gradient toward
        # favourable type matchups and to give switching a larger signal.
        matchup_score   = self._compute_matchup_score(self.ia_pokemon, self.rival_pokemon)
        matchup_shaping = 0.035 * matchup_score   # ±0.035 maximum

        # ── 10. Threat-level estimate ─────────────────────────────────────────
        # 0.0 = rival poses no meaningful threat; 1.0 = expected KO next hit.
        threat_level = self._estimate_threat_level(self.rival_pokemon, self.ia_pokemon)

        # ── 11. Bad-stay penalty ──────────────────────────────────────────────
        # Fires when matchup is clearly losing (score < −0.15) AND the IA did
        # NOT switch this turn.  Strengthened vs previous version:
        #   base:   −0.020 × |matchup_score|   (was −0.015)
        #   threat: −0.025 × threat_level       (was −0.020)
        # Combined max ≤ −0.045/turn in the worst matchup under full threat.
        bad_stay_penalty = 0.0
        if matchup_score < -0.15 and switch_quality_delta == 0.0:
            bad_stay_penalty = (
                -0.020 * abs(matchup_score)
                - 0.025 * threat_level
            )

        # ── 12. Temporal risk penalty ─────────────────────────────────────────
        # Independent of matchup: penalises staying in when the opponent can
        # likely KO the IA in one hit, regardless of type advantage.
        # Strengthened from −0.025 → −0.040 for a sharper deterrent signal.
        temporal_risk = -0.040 * threat_level

        # ── 13. Momentum / tempo reward ───────────────────────────────────────
        # EMA of (damage_dealt − damage_taken).  Positive = IA is controlling
        # the pace of the battle; negative = IA is being pressured.
        # Strengthened: max ±0.020 (was ±0.015), slightly slower decay (α=0.35)
        # so it reflects the past 3–4 turns rather than just the last turn.
        current_delta        = damage_to_rival - damage_to_ia
        self.damage_momentum = 0.35 * current_delta + 0.65 * self.damage_momentum
        momentum_reward      = float(0.020 * np.tanh(self.damage_momentum * 4.0))

        # ── 14. Move quality (effectiveness + accuracy) ───────────────────────
        # Teaches the PPO to prefer SE moves and avoid immunities / low-accuracy
        # moves that waste turns.  Unchanged from previous version.
        #
        #   immune (0×)  → −0.06  |  0.25× → −0.04  |  0.5× → −0.02
        #   1× neutral   →  0.00  |  2×    → +0.02   |  4×   → +0.04
        #   accuracy 50% adds −0.004 on top of effectiveness component
        move_quality = 0.0
        if ia_effectiveness is not None and ia_power > 0:
            if ia_effectiveness == 0.0:
                move_quality = -0.06
            else:
                eff_component = float(
                    np.clip(0.02 * np.log2(max(1e-9, float(ia_effectiveness))), -0.04, 0.04)
                )
                acc_penalty = -0.008 * max(0.0, 1.0 - float(ia_accuracy) / 100.0)
                move_quality = eff_component + acc_penalty

        # ── 15. Smart switch quality (switch turns only) ───────────────────────
        # Rewards switching into a better matchup / lower threat.
        # Anti-spam fatigue reduces reward after 8 switches (floor at 0.70×).
        switch_fatigue = min(1.0, self.switches_this_episode / 8.0)
        fatigue_mult   = max(0.70, 1.0 - 0.30 * switch_fatigue)
        smart_switch   = 0.06 * float(np.clip(switch_quality_delta, -1.0, 1.0)) * fatigue_mult

        # ── Assemble total reward ──────────────────────────────────────────────
        reward = (
            damage_reward
            + ko_reward
            + terminal_bonus
            + anti_burst_penalty
            + stall_penalty
            + survival_bonus
            + hp_lead_bonus
            + consecutive_bonus
            - switch_penalty
            + matchup_shaping
            + bad_stay_penalty
            + temporal_risk
            + momentum_reward
            + move_quality
            + smart_switch
        )

        self.episode_reward += reward
        self.last_reward_breakdown = {
            # Primary signals
            "reward":               reward,
            "terminal_bonus":       terminal_bonus,
            "anti_burst_penalty":   anti_burst_penalty,
            # KO signals
            "ko_bonus":             ko_bonus,
            "faint_penalty":       -faint_penalty,
            "ko_reward":            ko_reward,
            # Damage signals
            "damage_reward":        damage_reward,
            "damage_dealt_reward":  0.12 * damage_to_rival,
            "damage_taken_penalty":-0.12 * damage_to_ia,
            # Per-turn strategic bonuses
            "survival_bonus":       survival_bonus,
            "hp_lead_bonus":        hp_lead_bonus,
            "consecutive_bonus":    consecutive_bonus,
            "consecutive_advantage":self.consecutive_advantage,
            # Per-turn costs
            "stall_penalty":        stall_penalty,
            "is_true_stall":        is_true_stall,
            "switch_penalty":      -switch_penalty,
            # Matchup signals
            "matchup_score":        matchup_score,
            "matchup_shaping":      matchup_shaping,
            "bad_stay_penalty":     bad_stay_penalty,
            # Threat / temporal signals
            "threat_level":         threat_level,
            "temporal_risk":        temporal_risk,
            # Momentum signal
            "damage_momentum":      self.damage_momentum,
            "momentum_reward":      momentum_reward,
            # Move quality
            "move_quality":         move_quality,
            "ia_effectiveness":     ia_effectiveness if ia_effectiveness is not None else 1.0,
            "ia_accuracy":          ia_accuracy,
            # Switch quality
            "smart_switch":         smart_switch,
            "switch_quality_delta": switch_quality_delta,
            "switch_fatigue":       switch_fatigue,
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
        except Exception as exc:
            warnings.warn(
                f"[PokemonEnv] _write_live_log failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

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
