"""
battle_engine.py
~~~~~~~~~~~~~~~~
Game Logic Layer — Pokémon Showdown-style battle simulator.

THIS MODULE IS COMPLETELY ISOLATED FROM THE RL TRAINING ENVIRONMENT.
It is used only by the Streamlit dashboard and offline simulations.

It implements ALL battle mechanics that were deliberately excluded from
PokemonEnv to keep training stable:
  - Status conditions  (burn, poison, paralysis, sleep, freeze)
  - Weather effects    (rain, sun, sandstorm, hail)
  - Entry hazards      (stealth rock, spikes)

PPO compatibility
-----------------
BattleEngine._get_obs() delegates to obs_builder.build_obs_28(), which
produces the exact same 28-dimensional vector as PokemonEnv._get_obs().
PPO models can therefore be used for inference without any modification.

Interface compatibility
-----------------------
The public interface intentionally mirrors PokemonEnv's battle methods:
  configure_battle(), step(), switch_turn(), _get_obs()
  Properties: hp_ia, hp_rival, weather, hazards_ia, hazards_rival,
              ia_pokemon, rival_pokemon, turn_count

This allows the Streamlit dashboard to swap in BattleEngine in place of
PokemonEnv with minimal code changes.
"""

from __future__ import annotations

import sqlite3
import warnings

import numpy as np

try:
    from src.battle_utils import (
        STAT_NAME_MAP,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_multiplier,
    )
    from src.battle_mechanics import (
        check_status_skip,
        get_hazard_entry_damage,
        get_paralysis_speed_factor,
        get_status_chip_damage,
        get_weather_chip_damage,
        get_weather_damage_multiplier,
        try_apply_move_status,
        WEATHER_TURNS_DEFAULT,
    )
    from src.game_engine.obs_builder import build_obs_28
    from src.pokemon_forms import resolve_form, normalize_pokemon_name
except ImportError:
    from battle_utils import (
        STAT_NAME_MAP,
        apply_stat_stages,
        describe_effectiveness,
        format_name,
        get_type_multiplier,
    )
    from battle_mechanics import (
        check_status_skip,
        get_hazard_entry_damage,
        get_paralysis_speed_factor,
        get_status_chip_damage,
        get_weather_chip_damage,
        get_weather_damage_multiplier,
        try_apply_move_status,
        WEATHER_TURNS_DEFAULT,
    )
    from game_engine.obs_builder import build_obs_28
    from pokemon_forms import resolve_form, normalize_pokemon_name


# ─────────────────────────────────────────────────────────────────────────────
# ITEM SLUG EXTRACTOR
#
# pokemon["item"] can be ANY of:
#   None            — no held item (RL training env default)
#   str             — canonical slug, e.g. "charizardite-x"
#   dict            — rich display object from get_item_data():
#                     {"name": "Charizardite X", "sprite": "https://..."}
#
# The UI legitimately needs the dict (to render item["name"] and item["sprite"]).
# The mechanics layer (resolve_form / _apply_item_transforms) needs a plain slug.
# _item_slug() is the single safe bridge between the two.
# ─────────────────────────────────────────────────────────────────────────────

def _item_slug(item) -> str:
    """
    Extract the canonical item slug from any held-item representation.

    Handles all three item formats that coexist in the codebase:

      None  → ""
      str   → lowercased, stripped, spaces→hyphens
              (e.g. "Charizardite X" → "charizardite-x")
      dict  → reads ["name"] key first, then ["slug"] / ["key"] as fallbacks,
              then applies the same normalisation as the str path
              (e.g. {"name": "Charizardite X", "sprite": "..."} → "charizardite-x")

    Returns
    -------
    str — always a string, never raises, empty string for unknown input.

    This is the ONLY place in the codebase that reads item data for mechanics
    purposes.  Adding a new item representation only requires updating this
    function — all call sites remain unchanged.
    """
    if not item:
        return ""
    if isinstance(item, str):
        return item.lower().strip().replace(" ", "-")
    if isinstance(item, dict):
        raw = (
            item.get("name")
            or item.get("slug")
            or item.get("key")
            or ""
        )
        return str(raw).lower().strip().replace(" ", "-")
    # Unexpected type — degrade gracefully rather than crash.
    return str(item).lower().strip().replace(" ", "-")


class BattleEngine:
    """
    Showdown-style battle simulator for the Game Logic Layer.

    Exposes the same interface as PokemonEnv's live-battle mode so the
    Streamlit dashboard can use it as a drop-in replacement.  It adds
    ALL the battle mechanics that are intentionally absent from the
    training environment.
    """

    def __init__(
        self,
        team_ia:    list[dict] | None = None,
        team_rival: list[dict] | None = None,
        max_turns:  int = 40,
        seed:       int | None = None,
        log_to_db:  bool = False,
    ):
        """
        Parameters
        ----------
        team_ia, team_rival
            Optional 6-Pokémon team lists.  When supplied the engine owns
            deep-copies of both teams and manages active indices internally.
        max_turns
            Maximum turns before the episode is truncated.
        seed : int | None
            Seed for the engine's internal RNG.  Same seed → identical battle
            outcomes for identical action sequences (deterministic replay).
            Defaults to None (non-reproducible) for live dashboard play.
        log_to_db : bool
            When True, every step writes to the SQLite v_logs table.
            Defaults to False so scripted simulations / RL evaluation never
            incur I/O overhead or depend on a database being present.
            Set to True only from the Streamlit dashboard.
        """
        self.max_turns = max_turns
        # Seeded NumPy generator — the ONLY source of randomness in this class.
        # All random decisions (speed tie, damage roll, status chance) route
        # through this generator so battles are fully reproducible given a seed.
        self._rng     = np.random.default_rng(seed)
        self._log_to_db = log_to_db

        # Active Pokémon (set by _load_teams or configure_battle)
        self._ia_pokemon:    dict | None = None
        self._rival_pokemon: dict | None = None

        # ── Weather ────────────────────────────────────────────────────────
        self._weather:       str | None  = None
        self._weather_turns: int         = 0

        # ── Entry hazards (set of strings per side) ────────────────────────
        self._hazards_ia:    set = set()
        self._hazards_rival: set = set()

        # ── Episode tracking (mirrors PokemonEnv for reward parity) ───────
        self.turn_count:               int   = 0
        self.hp_ia:                    float = 1.0
        self.hp_rival:                 float = 1.0
        self._episode_reward:          float = 0.0
        self._episode_damage_dealt:    float = 0.0
        self._episode_damage_received: float = 0.0
        self._episode_kos_for:         int   = 0
        self._episode_kos_against:     int   = 0
        self._episode_stalled_turns:   int   = 0
        self._last_reward_breakdown:   dict  = {}
        self._damage_momentum:         float = 0.0   # EMA tempo signal
        self._switches_this_episode:   int   = 0     # anti-spam switch counter
        self._consecutive_advantage:   int   = 0     # turns with HP lead > 0.05

        # ── Team ownership ─────────────────────────────────────────────────
        # The engine holds deep copies of both teams and is the sole mutator
        # of their HP, status, stat_stages, and debilitado fields.
        # The UI must never mutate these dicts directly — read via get_state().
        self._team_ia:    list[dict] = []
        self._team_rival: list[dict] = []
        self._active_ia:   int = 0
        self._active_rival: int = 0

        if team_ia and team_rival:
            self._load_teams(team_ia, team_rival)

    # ── Read-only properties (dashboard-compatible interface) ──────────────

    @property
    def ia_pokemon(self) -> dict | None:
        return self._ia_pokemon

    @property
    def rival_pokemon(self) -> dict | None:
        return self._rival_pokemon

    @property
    def weather(self) -> str | None:
        return self._weather

    @property
    def hazards_ia(self) -> set:
        return self._hazards_ia

    @property
    def hazards_rival(self) -> set:
        return self._hazards_rival

    # ── Battle setup ───────────────────────────────────────────────────────

    def configure_battle(self, ia_pokemon: dict, rival_pokemon: dict) -> None:
        """
        Set the active Pokémon for the current turn.

        Called by the dashboard before every combat step and after a switch.
        Does NOT reset episode trackers — those reset only once per full
        battle via start_battle() or on engine creation.
        """
        self._ia_pokemon    = ia_pokemon
        self._rival_pokemon = rival_pokemon
        self._sync_pokemon_state(self._ia_pokemon)
        self._sync_pokemon_state(self._rival_pokemon)

    def start_battle(self) -> None:
        """
        Reset all episode trackers for a fresh battle.

        Not needed if a new BattleEngine instance is created per battle
        (which is how the dashboard manages it via session state), but
        useful for scripted simulations.
        """
        self.turn_count               = 0
        self.hp_ia                    = 1.0
        self.hp_rival                 = 1.0
        self._episode_reward          = 0.0
        self._episode_damage_dealt    = 0.0
        self._episode_damage_received = 0.0
        self._episode_kos_for         = 0
        self._episode_kos_against     = 0
        self._episode_stalled_turns   = 0
        self._last_reward_breakdown   = {}
        self._weather                 = None
        self._weather_turns           = 0
        self._hazards_ia              = set()
        self._hazards_rival           = set()
        # ── Advanced reward tracking (mirrors PokemonEnv._reset_episode_trackers) ──
        self._damage_momentum         = 0.0   # EMA of (damage_dealt − damage_taken)
        self._switches_this_episode   = 0     # voluntary IA switches, for anti-spam fatigue
        self._consecutive_advantage   = 0     # turns with meaningful HP lead > 0.05

    # ── Team ownership API ─────────────────────────────────────────────────
    #
    # These methods form the clean interface between the engine and the UI.
    # The engine is the SOLE MUTATOR of team state.  The UI reads via
    # get_state() and sends intents via step() / send_in() / switch_turn().
    # Pure Python only — zero Streamlit / session_state dependencies below.

    def _load_teams(self, team_ia: list[dict], team_rival: list[dict]) -> None:
        """
        Deep-copy both teams into engine ownership and reset all Pokémon to
        full battle-start state.  Called once at construction time.

        The original dicts passed by the caller are NEVER mutated — the engine
        works exclusively on its own copies.
        """
        import copy

        def _fresh(p: dict) -> dict:
            c = copy.deepcopy(p)
            c["current_hp"]   = 1.0
            c["status"]       = None
            c["debilitado"]   = False
            c["stat_stages"]  = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
            c["stats"]        = apply_stat_stages(c["base_stats"], c["stat_stages"])
            # ── Form / shiny state fields ────────────────────────────────────
            # species: canonical base-species slug (never changes mid-battle)
            raw_name = c.get("name", "")
            c.setdefault("species", normalize_pokemon_name(
                raw_name.lower().strip().replace(" ", "-")
            ))
            # form: current active form slug (updated by _apply_item_transforms)
            c.setdefault("form", c["species"])
            # shiny: cosmetic flag — rolled at team-creation time via seeded RNG
            c.setdefault("shiny", False)
            # mega_evolved: guard that prevents double-mega within one battle
            c["mega_evolved"] = False
            return c

        self._team_ia    = [_fresh(p) for p in team_ia]
        self._team_rival = [_fresh(p) for p in team_rival]
        self._active_ia    = 0
        self._active_rival = 0

        if self._team_ia:
            self._ia_pokemon = self._team_ia[0]
            self.hp_ia       = 1.0
        if self._team_rival:
            self._rival_pokemon = self._team_rival[0]
            self.hp_rival       = 1.0

    def get_state(self) -> dict:
        """
        Return the current battle state as a plain dict for UI rendering.

        The returned team lists are REFERENCES to the engine's internal
        deep-copied team dicts.  The UI must treat them as READ-ONLY — the
        engine is the sole mutator.  Reading them directly (not copying) means
        the UI always sees the live state without an extra sync step.

        Keys
        ----
        team_ia / team_rival   — list of 6 Pokémon dicts (engine-owned)
        active_ia / active_rival — int index of the currently active slot
        hp_ia / hp_rival       — float in [0, 1] for the active Pokémon
        weather                — str | None
        hazards_ia/rival       — frozenset of active hazard strings
        turn                   — int turn counter
        fainted_ia/rival       — list[bool] per-slot faint flags
        all_ia_fainted         — True when the IA has lost all Pokémon
        all_rival_fainted      — True when the rival has lost all Pokémon
        """
        fainted_ia    = [bool(p.get("debilitado", False)) for p in self._team_ia]
        fainted_rival = [bool(p.get("debilitado", False)) for p in self._team_rival]
        return {
            "team_ia":           self._team_ia,
            "team_rival":        self._team_rival,
            "active_ia":         self._active_ia,
            "active_rival":      self._active_rival,
            "hp_ia":             self.hp_ia,
            "hp_rival":          self.hp_rival,
            "weather":           self._weather,
            "hazards_ia":        frozenset(self._hazards_ia),
            "hazards_rival":     frozenset(self._hazards_rival),
            "turn":              self.turn_count,
            "fainted_ia":        fainted_ia,
            "fainted_rival":     fainted_rival,
            "all_ia_fainted":    all(fainted_ia) if fainted_ia    else False,
            "all_rival_fainted": all(fainted_rival) if fainted_rival else False,
        }

    def find_next_available(self, side: str) -> int | None:
        """
        Return the index of the first non-fainted Pokémon for *side*
        ("ia" or "rival"), or None if all have fainted.
        """
        team = self._team_ia if side == "ia" else self._team_rival
        for i, p in enumerate(team):
            if not p.get("debilitado", False):
                return i
        return None

    def send_in(self, side: str, idx: int) -> str:
        """
        Send in the Pokémon at position *idx* for *side*, applying any
        entry-hazard chip damage.

        Updates _active_ia/_active_rival and _ia_pokemon/_rival_pokemon.
        Returns a hazard log string (empty string if no hazards triggered).
        """
        if side == "ia":
            team    = self._team_ia
            hazards = self._hazards_ia
        else:
            team    = self._team_rival
            hazards = self._hazards_rival

        pokemon = team[idx]
        pokemon["current_hp"] = max(0.0, float(pokemon.get("current_hp", 1.0)))

        haz_log = ""
        if hazards:
            chip, haz_log = get_hazard_entry_damage(pokemon, hazards)
            if chip > 0:
                pokemon["current_hp"] = max(0.0, pokemon["current_hp"] - chip)

        if side == "ia":
            self._active_ia  = idx
            self._ia_pokemon = pokemon
            self._sync_pokemon_state(self._ia_pokemon)
        else:
            self._active_rival  = idx
            self._rival_pokemon = pokemon
            self._sync_pokemon_state(self._rival_pokemon)

        return haz_log

    def handle_post_faint(
        self,
        side:           str,
        challenge_mode: bool = False,
    ) -> dict:
        """
        Process the aftermath of a Pokémon fainting on *side*.

        In simulation mode (challenge_mode=False, or side=="ia"):
            Automatically sends in the next available Pokémon.
        In challenge mode (challenge_mode=True, side=="rival"):
            Signals the UI that the player must choose their next Pokémon.

        Returns
        -------
        dict with keys:
            battle_over   — True when all Pokémon on this side have fainted
            outcome       — human-readable result string (non-empty if over)
            auto_switched — True when the engine automatically sent in a new one
            next_idx      — int index of the new active Pokémon, or None
            haz_log       — entry-hazard log string from the automatic switch-in
            must_choose   — True when the player must manually pick next
        """
        # debilitado is already set by _execute_move → _sync_pokemon_state;
        # we just need to find the next available slot.
        next_idx = self.find_next_available(side)

        if next_idx is None:
            outcome = (
                "🏆 ¡VICTORIA DE LA IA!" if side == "rival"
                else "💀 LA IA HA SIDO DERROTADA"
            )
            return {
                "battle_over": True,  "outcome": outcome,
                "auto_switched": False, "next_idx": None,
                "haz_log": "",        "must_choose": False,
            }

        # Challenge mode: rival player must choose manually
        if challenge_mode and side == "rival":
            return {
                "battle_over": False,  "outcome": "",
                "auto_switched": False, "next_idx": next_idx,
                "haz_log": "",         "must_choose": True,
            }

        # Simulation (auto-advance) or IA side always auto-advances
        haz_log = self.send_in(side, next_idx)
        return {
            "battle_over": False,  "outcome": "",
            "auto_switched": True, "next_idx": next_idx,
            "haz_log": haz_log,    "must_choose": False,
        }

    # ── PPO inference bridge ───────────────────────────────────────────────

    def _get_obs(self, for_rival: bool = False) -> np.ndarray:
        """
        Return the 28-dim observation vector compatible with PokemonEnv.

        This is the bridge between BattleEngine and any PPO model trained
        on PokemonEnv.  It delegates to obs_builder.build_obs_28() which
        implements the SHARED observation contract.
        """
        me  = self._rival_pokemon if for_rival else self._ia_pokemon
        foe = self._ia_pokemon    if for_rival else self._rival_pokemon
        if me is None or foe is None:
            raise RuntimeError("configure_battle() must be called before _get_obs()")
        return build_obs_28(me, foe)

    # ── Battle execution ───────────────────────────────────────────────────

    def step(
        self,
        action_ia:    int,
        action_rival: int | None = None,
        ia_move_name: str | None = None,   # unused, kept for interface parity
    ) -> tuple:
        """
        Execute one full turn.

        Returns (obs, reward, terminated, truncated, info) — identical
        format to PokemonEnv.step().
        """
        if self._ia_pokemon is None or self._rival_pokemon is None:
            raise RuntimeError(
                "Active Pokémon not set — call configure_battle() or pass "
                "team_ia/team_rival to BattleEngine() before step()."
            )

        action_ia = self._normalize_action(action_ia, self._ia_pokemon)
        if action_rival is None:
            action_rival = self._select_opponent_action()
        else:
            action_rival = self._normalize_action(action_rival, self._rival_pokemon)

        return self._run_turn(action_ia, action_rival)

    def switch_turn(
        self,
        side:               str,
        new_active_pokemon: dict,
        opponent_action:    int | None = None,
    ) -> tuple:
        """
        Handle a voluntary Pokémon switch.

        Returns (obs, reward, terminated, truncated, info) — same format
        as PokemonEnv.switch_turn().
        """
        reward = 0.0

        if side == "rival":
            old_name    = self._rival_pokemon["name"]
            self._rival_pokemon = new_active_pokemon
            self._sync_pokemon_state(self._rival_pokemon)
            # Keep active-index in sync with engine-owned team list
            try:
                self._active_rival = self._team_rival.index(new_active_pokemon)
            except ValueError:
                pass  # team list not yet populated (backward-compat path)
            switch_log   = f"{old_name} switched out for {new_active_pokemon['name']}"
            attack_result = None
            old_hp_ia    = self.hp_ia
            old_hp_rival = self.hp_rival

            if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
                ia_action = self._normalize_action(opponent_action, self._ia_pokemon)
                attack_result = self._execute_move(
                    self._ia_pokemon,
                    self._rival_pokemon,
                    self._ia_pokemon["moves"][ia_action],
                )
                reward_data = self._compute_reward(old_hp_ia, old_hp_rival, switch_penalty=0.0)
                reward = reward_data["reward"]

            self._write_live_log(
                ia_result=attack_result,
                rival_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            )
            info = self._build_turn_info(
                reward=reward, reward_data=self._last_reward_breakdown,
                old_hp_ia=old_hp_ia, old_hp_rival=old_hp_rival,
                ia_result=attack_result,
                rival_result={"log": switch_log, "type": "", "effectiveness_label": ""},
                switch_log=switch_log,
            )
            return self._get_obs(), reward, self._is_battle_over(), False, info

        # side == "ia"
        # Capture pre-switch matchup AND threat before overwriting self._ia_pokemon
        old_ia_pokemon = self._ia_pokemon
        old_matchup    = self._compute_matchup_score(old_ia_pokemon, self._rival_pokemon)
        old_threat     = self._estimate_threat_level(self._rival_pokemon, old_ia_pokemon)

        old_name         = old_ia_pokemon["name"]
        self._ia_pokemon = new_active_pokemon
        self._sync_pokemon_state(self._ia_pokemon)
        # Keep active-index in sync with engine-owned team list
        try:
            self._active_ia = self._team_ia.index(new_active_pokemon)
        except ValueError:
            pass  # team list not yet populated (backward-compat path)

        new_matchup  = self._compute_matchup_score(new_active_pokemon, self._rival_pokemon)
        new_threat   = self._estimate_threat_level(self._rival_pokemon, new_active_pokemon)

        # Combined delta: matchup improvement + 0.5 × threat reduction
        switch_quality_delta = (new_matchup - old_matchup) + 0.5 * (old_threat - new_threat)

        # Track for anti-spam fatigue
        self._switches_this_episode += 1

        switch_log   = f"{old_name} switched out for {new_active_pokemon['name']}"
        old_hp_ia    = self.hp_ia
        old_hp_rival = self.hp_rival
        attack_result = None

        if opponent_action is not None and self.hp_ia > 0 and self.hp_rival > 0:
            rival_action = self._normalize_action(opponent_action, self._rival_pokemon)
            attack_result = self._execute_move(
                self._rival_pokemon,
                self._ia_pokemon,
                self._rival_pokemon["moves"][rival_action],
            )

        reward_data = self._compute_reward(
            old_hp_ia, old_hp_rival,
            switch_penalty=0.03,
            switch_quality_delta=switch_quality_delta,
        )
        reward = reward_data["reward"]
        self._write_live_log(
            ia_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            rival_result=attack_result,
        )
        info = self._build_turn_info(
            reward=reward, reward_data=reward_data,
            old_hp_ia=old_hp_ia, old_hp_rival=old_hp_rival,
            ia_result={"log": switch_log, "type": "", "effectiveness_label": ""},
            rival_result=attack_result,
            switch_log=switch_log,
        )
        return self._get_obs(), reward, self._is_battle_over(), False, info

    # ── Item-triggered form transforms ────────────────────────────────────

    def _apply_item_transforms(self, pokemon: dict) -> str | None:
        """
        Apply held-item triggered form transformation (Mega Evolution, G-Max).

        Called at the START of each turn for both active Pokémon.

        DESIGN CONTRACT
        ───────────────
        • Deterministic: depends only on ``pokemon["item"]`` and lookup table.
        • Idempotent: ``mega_evolved`` guard prevents double transformation.
        • No randomness: form resolution is a pure table lookup via resolve_form().
        • Stat scaling: approximated by stat_mult from MEGA_STONE_MAP.  Exact
          per-stat Mega boosts vary per species; this is a faithful approximation
          for RL training and is consistent between UI and environment.
        • HP: never changed by Mega — only G-Max / Dynamax double HP (hp_mult=2).
          We apply hp_mult only once, guarded by mega_evolved.
        • Type override: Mega forms that change typing (e.g. Charizard-Mega-X
          gains Dragon, Aggron-Mega drops Rock) are applied from resolve_form().

        Parameters
        ----------
        pokemon : dict
            Live Pokémon state dict (engine-owned deep copy).

        Returns
        -------
        str | None
            Human-readable transform log line, or None if no transform occurred.
        """
        # Guard: already transformed this battle
        if pokemon.get("mega_evolved", False):
            return None

        # _item_slug() handles None / str / dict uniformly — never raises.
        item = _item_slug(pokemon.get("item"))
        if not item:
            return None

        species = pokemon.get("species") or normalize_pokemon_name(
            pokemon.get("name", "").lower().strip().replace(" ", "-")
        )

        form_info = resolve_form(species, item)
        if form_info["form_type"] == "base":
            return None  # non-transform item (Life Orb, Leftovers, …)

        # ── Apply transform ──────────────────────────────────────────────────
        old_form              = pokemon.get("form", species)
        pokemon["form"]       = form_info["form_name"]
        pokemon["mega_evolved"] = True   # prevent re-triggering next turn

        # Type override (e.g. Charizard-Mega-X → ["fire", "dragon"])
        if form_info.get("types"):
            pokemon["types"] = list(form_info["types"])

        # Stat scaling (approximate; consistent with MEGA_STONE_MAP entries)
        stat_mult = form_info.get("stat_mult", 1.0)
        if stat_mult != 1.0:
            base = pokemon.get("base_stats", pokemon.get("stats", {}))
            new_stats = {k: max(1, int(v * stat_mult)) for k, v in base.items()}
            pokemon["stats"] = new_stats

        # HP multiplier (G-Max / Dynamax only — mega_evolved guard ensures once)
        hp_mult = form_info.get("hp_mult", 1.0)
        if hp_mult != 1.0:
            pokemon["current_hp"] = min(1.0, float(pokemon.get("current_hp", 1.0)) * hp_mult)

        form_label = {
            "mega":    "Mega Evolved",
            "gmax":    "Gigantamaxed",
            "dynamax": "Dynamaxed",
        }.get(form_info["form_type"], "transformed")

        return (
            f"✨ {pokemon.get('name', species)} {form_label}! "
            f"({old_form} → {pokemon['form']})"
        )

    # ── Turn execution (FULL mechanics — status, weather, hazards) ─────────

    def _run_turn(self, action_ia: int, action_rival: int) -> tuple:
        self.turn_count += 1
        old_hp_ia    = float(self._ia_pokemon.get("current_hp",    1.0))
        old_hp_rival = float(self._rival_pokemon.get("current_hp", 1.0))

        # ── Item-triggered form transforms (start of turn, once per battle) ─
        transform_log_ia    = self._apply_item_transforms(self._ia_pokemon)
        transform_log_rival = self._apply_item_transforms(self._rival_pokemon)

        ia_move    = self._ia_pokemon["moves"][action_ia]
        rival_move = self._rival_pokemon["moves"][action_rival]

        ia_result    = None
        rival_result = None

        # ── Speed with paralysis modifier ──────────────────────────────────
        ia_speed    = self._ia_pokemon["stats"].get("spd",    1) * get_paralysis_speed_factor(self._ia_pokemon)
        rival_speed = self._rival_pokemon["stats"].get("spd", 1) * get_paralysis_speed_factor(self._rival_pokemon)
        ia_first = ia_speed >= rival_speed if ia_speed != rival_speed else bool(self._rng.random() < 0.5)

        # ── Status block check ─────────────────────────────────────────────
        ia_blocked,    ia_block_log    = check_status_skip(self._ia_pokemon)
        rival_blocked, rival_block_log = check_status_skip(self._rival_pokemon)

        def _skip(log: str) -> dict:
            return {"log": log, "type": "", "effectiveness": 1.0, "effectiveness_label": ""}

        if ia_blocked:
            ia_result = _skip(ia_block_log)
        if rival_blocked:
            rival_result = _skip(rival_block_log)

        # ── Move execution ─────────────────────────────────────────────────
        if ia_first:
            if not ia_blocked:
                ia_result = self._execute_move(self._ia_pokemon, self._rival_pokemon, ia_move)
            if self._rival_pokemon.get("current_hp", 0.0) > 0 and not rival_blocked:
                rival_result = self._execute_move(self._rival_pokemon, self._ia_pokemon, rival_move)
        else:
            if not rival_blocked:
                rival_result = self._execute_move(self._rival_pokemon, self._ia_pokemon, rival_move)
            if self._ia_pokemon.get("current_hp", 0.0) > 0 and not ia_blocked:
                ia_result = self._execute_move(self._ia_pokemon, self._rival_pokemon, ia_move)

        # ── End-of-turn effects (burn, poison, weather chip) ───────────────
        self._apply_end_turn_effects()

        # Pass IA move quality context (mirrors PokemonEnv._run_turn)
        ia_eff = ia_result.get("effectiveness") if ia_result else None
        ia_pwr = ia_move.get("power") or 0
        ia_acc = ia_move.get("accuracy") or 100
        reward_data = self._compute_reward(
            old_hp_ia, old_hp_rival,
            ia_effectiveness=ia_eff,
            ia_power=ia_pwr,
            ia_accuracy=ia_acc,
        )
        reward = reward_data["reward"]

        self._write_live_log(ia_result=ia_result, rival_result=rival_result)

        terminated = self._is_battle_over()
        truncated  = self.turn_count >= self.max_turns
        info = self._build_turn_info(
            reward=reward, reward_data=reward_data,
            old_hp_ia=old_hp_ia, old_hp_rival=old_hp_rival,
            ia_result=ia_result, rival_result=rival_result,
        )
        # Surface any form-transform events for the UI to display
        if transform_log_ia:
            info["transform_ia"] = transform_log_ia
        if transform_log_rival:
            info["transform_rival"] = transform_log_rival
        if truncated and not terminated:
            info["is_win"] = self.hp_rival < self.hp_ia
        return self._get_obs(), reward, terminated, truncated, info

    def _execute_move(self, attacker: dict, defender: dict, move: dict) -> dict:
        """
        Execute one move with full Game Logic Layer mechanics:
          - Burn halves physical attack
          - Weather multiplies relevant move types
          - Status conditions can be inflicted by moves
        """
        move_type    = move.get("type", "normal")
        effectiveness = get_type_multiplier(move_type, defender.get("types", []))
        damage_class  = move.get("damage_class", "status")
        power         = move.get("power") or 0

        # ── Status / zero-power moves ──────────────────────────────────────
        if damage_class == "status" or power == 0:
            self._apply_move_effects(attacker, defender, move)
            status_log = try_apply_move_status(move, defender)
            log = f"{attacker['name']} used {move['name']} ({format_name(move_type)})"
            if move.get("stat_changes"):
                log += f" [{self._format_stat_changes(move)}]"
            if status_log:
                log += f" | {status_log}"
            return {
                "log": log,
                "type": format_name(move_type),
                "effectiveness": effectiveness,
                "effectiveness_label": describe_effectiveness(effectiveness),
                "status_applied": status_log,
            }

        # ── Damaging moves ─────────────────────────────────────────────────
        attack_key  = "atk" if damage_class == "physical" else "sp_atk"
        defense_key = "def" if damage_class == "physical" else "sp_def"

        attack_stat = max(1, attacker["stats"].get(attack_key, 1))
        # Burn halves physical attack (Gen III+ mechanics)
        if damage_class == "physical" and attacker.get("status") == "burn":
            attack_stat = max(1, int(attack_stat * 0.5))

        defense_stat  = max(1, defender["stats"].get(defense_key, 1))
        stab          = 1.5 if move_type in attacker.get("types", []) else 1.0
        weather_mod   = get_weather_damage_multiplier(move_type, self._weather)
        random_factor = float(self._rng.uniform(0.92, 1.0))

        damage = (
            ((22 * power * attack_stat / defense_stat) / 50) + 2
        ) * stab * effectiveness * weather_mod * random_factor

        damage_ratio = (
            0.0 if effectiveness == 0
            else max(0.01, damage / max(1, defender["base_stats"]["hp"] * 2.5))
        )
        # NOTE: no upper cap — super effective / STAB hits can now OHKO,
        # which is correct Pokémon behaviour.  The old min(0.95, …) was an
        # RL training guard that caused all lethal hits to leave exactly 5% HP.

        defender["current_hp"] = float(
            np.clip(defender.get("current_hp", 1.0) - damage_ratio, 0.0, 1.0)
        )
        defender["debilitado"] = defender["current_hp"] <= 0
        self._sync_pokemon_state(attacker)
        self._sync_pokemon_state(defender)

        # Secondary status effect (e.g. Scald → burn, Thunderbolt → paralysis)
        status_log = try_apply_move_status(move, defender)
        log = (
            f"{attacker['name']} used {move['name']} "
            f"({format_name(move_type)}) [{describe_effectiveness(effectiveness)}]"
        )
        if status_log:
            log += f" | {status_log}"

        return {
            "log": log,
            "type": format_name(move_type),
            "effectiveness": effectiveness,
            "effectiveness_label": describe_effectiveness(effectiveness),
            "status_applied": status_log,
        }

    def _apply_end_turn_effects(self) -> None:
        """
        End-of-turn chip damage from:
          - burn   (1/16 max HP per turn)
          - poison  (1/8 max HP per turn)
          - sandstorm / hail weather (1/16 max HP, with type immunities)
        """
        for pokemon in (self._ia_pokemon, self._rival_pokemon):
            if pokemon is None or pokemon.get("debilitado"):
                continue
            chip_status,  _ = get_status_chip_damage(pokemon)
            chip_weather, _ = get_weather_chip_damage(pokemon, self._weather)
            total = chip_status + chip_weather
            if total > 0:
                pokemon["current_hp"] = max(
                    0.0, float(pokemon.get("current_hp", 1.0)) - total
                )
                self._sync_pokemon_state(pokemon)

        # Tick weather down
        if self._weather and self._weather_turns > 0:
            self._weather_turns -= 1
            if self._weather_turns == 0:
                self._weather = None

    # ── Weather management (called by UI or move effects) ──────────────────

    def set_weather(self, weather: str | None, turns: int | None = None) -> None:
        """
        Set or clear weather.  Accepts None to clear.
        Defaults to WEATHER_TURNS_DEFAULT turns if not specified.
        """
        self._weather = weather
        self._weather_turns = (turns if turns is not None else WEATHER_TURNS_DEFAULT) if weather else 0

    # ── State management ───────────────────────────────────────────────────

    def _sync_pokemon_state(self, pokemon: dict) -> None:
        pokemon["stats"] = apply_stat_stages(
            pokemon["base_stats"], pokemon.get("stat_stages", {})
        )
        pokemon["current_hp"] = float(np.clip(pokemon.get("current_hp", 1.0), 0.0, 1.0))
        pokemon["debilitado"]  = pokemon["current_hp"] <= 0
        if pokemon is self._ia_pokemon:
            self.hp_ia    = float(pokemon.get("current_hp", self.hp_ia))
        elif pokemon is self._rival_pokemon:
            self.hp_rival = float(pokemon.get("current_hp", self.hp_rival))

    def _is_battle_over(self) -> bool:
        return self.hp_ia <= 0 or self.hp_rival <= 0

    def _normalize_action(self, action, pokemon: dict) -> int:
        if hasattr(action, "item"):
            action = int(action.item())
        move_count = max(1, len(pokemon.get("moves", [])))
        return max(0, min(int(action), move_count - 1))

    def _select_opponent_action(self) -> int:
        """Random action for live-battle opponent (player or auto)."""
        move_total = max(1, len(self._rival_pokemon.get("moves", [])))
        return int(self._rng.integers(0, min(4, move_total)))

    def _apply_move_effects(self, attacker: dict, defender: dict, move: dict) -> None:
        changes = move.get("stat_changes", [])
        target  = (move.get("target") or "").lower()
        target_pokemon = (
            attacker if target in {"user", "users-field", "entire-field", "ally"} else defender
        )
        for change in changes:
            stat_name = STAT_NAME_MAP.get(change.get("name"))
            if not stat_name:
                continue
            target_pokemon["stat_stages"][stat_name] = max(
                -6, min(6, target_pokemon["stat_stages"].get(stat_name, 0) + change.get("change", 0))
            )
        self._sync_pokemon_state(attacker)
        self._sync_pokemon_state(defender)

    def _format_stat_changes(self, move: dict) -> str:
        parts = []
        for change in move.get("stat_changes", []):
            stat_name = STAT_NAME_MAP.get(change.get("name"), change.get("name", "stat"))
            delta = change.get("change", 0)
            if delta:
                parts.append(f"{stat_name} {delta:+d}")
        return ", ".join(parts)

    # ── Reward helpers ─────────────────────────────────────────────────────────

    def _compute_matchup_score(self, me: dict, foe: dict) -> float:
        """Signed matchup quality in [−1, +1] (mirrors PokemonEnv._compute_matchup_score)."""
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

    def _estimate_threat_level(self, attacker: dict, defender: dict) -> float:
        """
        Heuristic KO-threat score in [0, 1] (mirrors PokemonEnv._estimate_threat_level).

        0.0 = no meaningful threat this turn
        1.0 = expected KO on the next hit
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
            move_type     = move.get("type", "normal")
            effectiveness = get_type_multiplier(move_type, defender.get("types", []))
            if effectiveness == 0:
                continue
            damage_class = move.get("damage_class", "physical")
            attack_key   = "atk" if damage_class == "physical" else "sp_atk"
            defense_key  = "def" if damage_class == "physical" else "sp_def"
            atk = max(1, attacker["stats"].get(attack_key, 1))
            dfn = max(1, defender["stats"].get(defense_key, 1))
            stab  = 1.5 if move_type in attacker.get("types", []) else 1.0
            dmg   = (((22 * power * atk / dfn) / 50) + 2) * stab * effectiveness * 0.96
            ratio = float(np.clip(dmg / max(1, defender["base_stats"]["hp"] * 2.5), 0.0, 0.95))
            best_dmg_ratio = max(best_dmg_ratio, ratio)

        if best_dmg_ratio <= 0.0:
            return 0.0

        turns_to_ko = current_hp / best_dmg_ratio
        return float(np.clip(1.0 - (turns_to_ko - 1.0) / 1.5, 0.0, 1.0))

    # ── Reward (informational — matches PokemonEnv formula exactly) ───────────

    def _compute_reward(
        self,
        old_hp_ia:            float,
        old_hp_rival:         float,
        switch_penalty:       float = 0.0,
        ia_effectiveness:     float | None = None,
        ia_power:             int   = 0,
        ia_accuracy:          int   = 100,
        switch_quality_delta: float = 0.0,
    ) -> dict:
        """
        15-component strategic reward — exact mirror of PokemonEnv._compute_reward.

        See PokemonEnv._compute_reward docstring for full design rationale.

        Components
        ----------
         1. damage_reward        ±0.12   — HP differential (reduced from 0.35)
         2. ko_reward            ±0.22   — KO / faint events (reduced from 0.35)
         3. terminal_bonus       ±1.00   — battle outcome (dominant signal)
         4. anti_burst_penalty   −0.15   — discount for wins in ≤ 4 turns
         5. stall_penalty        −0.015  — ONLY on true stalls (0 damage both sides)
         6. survival_bonus       +0.008  — per turn alive (accumulates)
         7. hp_lead_bonus        +0.015  — proportional to HP advantage
         8. consecutive_bonus    0..+0.03— compound sustained-lead reward
         9. matchup_shaping      ±0.035  — continuous type-advantage gradient
        10. bad_stay_penalty     ≤−0.045 — losing matchup × threat amplifier
        11. temporal_risk        ≤−0.040 — per-turn KO-threat deterrent
        12. momentum_reward      ±0.020  — EMA of damage differential
        13. move_quality         −0.06..+0.04 — SE/immunity + accuracy feedback
        14. smart_switch         ±0.06   — quality delta with anti-spam fatigue
        15. switch_cost          var     — base friction on IA switches
        """
        damage_to_rival = max(0.0, old_hp_rival - self.hp_rival)
        damage_to_ia    = max(0.0, old_hp_ia    - self.hp_ia)
        self._episode_damage_dealt    += damage_to_rival
        self._episode_damage_received += damage_to_ia
        is_true_stall = damage_to_rival <= 1e-9 and damage_to_ia <= 1e-9
        if is_true_stall:
            self._episode_stalled_turns += 1

        # 1. Damage differential (reduced weight)
        damage_reward = 0.12 * damage_to_rival - 0.12 * damage_to_ia

        # 2. KO / faint (reduced weight)
        ko_bonus  = 0.22 if old_hp_rival > 0 and self.hp_rival <= 0 else 0.0
        faint_pen = 0.22 if old_hp_ia    > 0 and self.hp_ia    <= 0 else 0.0
        if ko_bonus  > 0: self._episode_kos_for     += 1
        if faint_pen > 0: self._episode_kos_against += 1
        ko_reward = ko_bonus - faint_pen

        # 3. Terminal bonus (unchanged)
        terminal_bonus = 0.0
        if   self.hp_rival <= 0: terminal_bonus =  1.0
        elif self.hp_ia    <= 0: terminal_bonus = -1.0

        # 4. Anti-burst (fast-win discount)
        anti_burst_penalty = 0.0
        if terminal_bonus > 0 and self.turn_count <= 4:
            anti_burst_penalty = -0.15

        # 5. Stall penalty (conditional — only true stalls)
        stall_penalty = -0.015 if is_true_stall else 0.0

        # 6. Survival bonus
        survival_bonus = 0.008 if self.hp_ia > 0 else 0.0

        # 7. HP-lead bonus
        hp_lead_bonus = 0.015 * max(0.0, self.hp_ia - self.hp_rival)

        # 8. Consecutive-advantage bonus
        if self.hp_ia > self.hp_rival + 0.05:
            self._consecutive_advantage = min(self._consecutive_advantage + 1, 10)
        else:
            self._consecutive_advantage = max(self._consecutive_advantage - 1, 0)
        consecutive_bonus = 0.003 * self._consecutive_advantage

        # 9. Matchup shaping (strengthened)
        matchup_score   = self._compute_matchup_score(self._ia_pokemon, self._rival_pokemon)
        matchup_shaping = 0.035 * matchup_score

        # 10. Threat level
        threat_level = self._estimate_threat_level(self._rival_pokemon, self._ia_pokemon)

        # 11. Bad-stay penalty (strengthened)
        bad_stay_penalty = 0.0
        if matchup_score < -0.15 and switch_quality_delta == 0.0:
            bad_stay_penalty = (
                -0.020 * abs(matchup_score)
                - 0.025 * threat_level
            )

        # 12. Temporal risk (strengthened)
        temporal_risk = -0.040 * threat_level

        # 13. Momentum EMA (strengthened, slower decay)
        current_delta          = damage_to_rival - damage_to_ia
        self._damage_momentum  = 0.35 * current_delta + 0.65 * self._damage_momentum
        momentum_reward        = float(0.020 * np.tanh(self._damage_momentum * 4.0))

        # 14. Move quality (effectiveness + accuracy)
        move_quality = 0.0
        if ia_effectiveness is not None and ia_power > 0:
            if ia_effectiveness == 0.0:
                move_quality = -0.06
            else:
                eff_component = float(
                    np.clip(0.02 * np.log2(max(1e-9, float(ia_effectiveness))), -0.04, 0.04)
                )
                acc_penalty  = -0.008 * max(0.0, 1.0 - float(ia_accuracy) / 100.0)
                move_quality = eff_component + acc_penalty

        # 15. Smart switch with anti-spam fatigue
        switch_fatigue = min(1.0, self._switches_this_episode / 8.0)
        fatigue_mult   = max(0.70, 1.0 - 0.30 * switch_fatigue)
        smart_switch   = 0.06 * float(np.clip(switch_quality_delta, -1.0, 1.0)) * fatigue_mult

        # Assemble
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

        self._episode_reward += reward
        self._last_reward_breakdown = {
            "reward":                reward,
            "terminal_bonus":        terminal_bonus,
            "anti_burst_penalty":    anti_burst_penalty,
            "ko_bonus":              ko_bonus,
            "faint_penalty":        -faint_pen,
            "ko_reward":             ko_reward,
            "damage_reward":         damage_reward,
            "damage_dealt_reward":   0.12 * damage_to_rival,
            "damage_taken_penalty": -0.12 * damage_to_ia,
            "survival_bonus":        survival_bonus,
            "hp_lead_bonus":         hp_lead_bonus,
            "consecutive_bonus":     consecutive_bonus,
            "consecutive_advantage": self._consecutive_advantage,
            "stall_penalty":         stall_penalty,
            "is_true_stall":         is_true_stall,
            "switch_penalty":       -switch_penalty,
            "matchup_score":         matchup_score,
            "matchup_shaping":       matchup_shaping,
            "bad_stay_penalty":      bad_stay_penalty,
            "threat_level":          threat_level,
            "temporal_risk":         temporal_risk,
            "damage_momentum":       self._damage_momentum,
            "momentum_reward":       momentum_reward,
            "move_quality":          move_quality,
            "ia_effectiveness":      ia_effectiveness if ia_effectiveness is not None else 1.0,
            "ia_accuracy":           ia_accuracy,
            "smart_switch":          smart_switch,
            "switch_quality_delta":  switch_quality_delta,
            "switch_fatigue":        switch_fatigue,
        }
        return self._last_reward_breakdown

    def _build_turn_info(
        self,
        reward,
        reward_data,
        old_hp_ia,
        old_hp_rival,
        ia_result,
        rival_result,
        switch_log: str = "",
    ) -> dict:
        terminated = self._is_battle_over()
        info = {
            "ia_move":             ia_result["log"]    if ia_result    else "Skipped",
            "rival_move":          rival_result["log"] if rival_result else "Skipped",
            "ia_move_type":        ia_result["type"]   if ia_result    else "",
            "rival_move_type":     rival_result["type"] if rival_result else "",
            "ia_effectiveness":    ia_result["effectiveness_label"]    if ia_result    else "",
            "rival_effectiveness": rival_result["effectiveness_label"] if rival_result else "",
            "hp_change_ia":        old_hp_ia    - self.hp_ia,
            "hp_change_rival":     old_hp_rival - self.hp_rival,
            "turn":                self.turn_count,
            "reward":              reward,
            "reward_breakdown":    reward_data,
            "damage_dealt":        self._episode_damage_dealt,
            "damage_received":     self._episode_damage_received,
            "ko_count":            self._episode_kos_for,
            "ko_received":         self._episode_kos_against,
            "stalled_turns":       self._episode_stalled_turns,
            "switch_log":          switch_log,
            "is_win":              terminated and self.hp_rival <= 0,
        }
        if terminated:
            info.update({
                "episode_reward":  self._episode_reward,
                "episode_length":  self.turn_count,
                "final_hp_ia":     self.hp_ia,
                "final_hp_rival":  self.hp_rival,
                "ko_count":        self._episode_kos_for,
                "ko_received":     self._episode_kos_against,
                "stalled_turns":   self._episode_stalled_turns,
            })
        return info

    # ── Logging ────────────────────────────────────────────────────────────

    def _write_live_log(
        self,
        ia_result:    dict | None,
        rival_result: dict | None,
    ) -> None:
        """Write turn data to SQLite.  No-op unless log_to_db=True was set at construction."""
        if not self._log_to_db:
            return
        try:
            conn = sqlite3.connect("pokemon_bigdata.db")
            cur  = conn.cursor()
            cur.execute(
                """
                INSERT INTO v_logs (
                    ia_move_name, rival_move, ia_move_type, rival_move_type,
                    ia_effectiveness, rival_effectiveness, hp_ia, hp_rival, reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ia_result["log"]    if ia_result    else "Skipped",
                    rival_result["log"] if rival_result else "Skipped",
                    ia_result["type"]   if ia_result    else "",
                    rival_result["type"] if rival_result else "",
                    ia_result["effectiveness_label"]    if ia_result    else "",
                    rival_result["effectiveness_label"] if rival_result else "",
                    self.hp_ia,
                    self.hp_rival,
                    self._last_reward_breakdown.get("reward", 0.0),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            warnings.warn(
                f"[BattleEngine] _write_live_log failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
