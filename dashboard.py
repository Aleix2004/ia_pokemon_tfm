import os
import random
import sqlite3
import time

import pandas as pd
import requests
import streamlit as st

from src.battle_utils import (
    apply_stat_stages,
    build_type_chart_rows,
    describe_effectiveness,
    format_name,
    get_type_multiplier,
)
from src.battle_mechanics import get_hazard_entry_damage
# ── Layer boundary: the dashboard uses the Game Engine layer, NOT the ──────
# ── training environment.  PokemonEnv is never imported here.          ──────
from src.game_engine.battle_engine import BattleEngine
from src.model_compat import check_model_compatibility, require_compatible_model
from src.type_colors import (
    get_type_colors,
    get_type_emoji,
    hp_bar_color,
    status_badge_html,
    type_badge_html,
    weather_badge_html,
)
from src.ai_advisor import get_greedy_action, get_hybrid_action
from src.competitive_movesets import (
    build_moveset,
    get_filtered_move_pool,
    get_role_info,
    prefilter_move_names,
)


st.set_page_config(layout="wide", page_title="Pokemon AI TFM Dashboard", page_icon="🧪")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def format_catalog_label(value):
    return " ".join(part.capitalize() for part in str(value).replace("-", " ").strip().split())


@st.cache_data
def get_pokemon_catalog():
    fallback = ["Mewtwo", "Rayquaza", "Kyogre", "Groudon", "Metagross", "Sceptile", "Charizard", "Blastoise", "Venusaur"]
    try:
        response = requests.get("https://pokeapi.co/api/v2/pokemon?limit=1500", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [format_catalog_label(entry["name"]) for entry in payload.get("results", [])]
    except Exception:
        return sorted(set(fallback))


@st.cache_data
def get_item_catalog():
    fallback = ["Life Orb", "Leftovers", "Choice Band", "Choice Specs", "Focus Sash", "Assault Vest"]
    try:
        response = requests.get("https://pokeapi.co/api/v2/item?limit=2500", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [format_catalog_label(entry["name"]) for entry in payload.get("results", [])]
    except Exception:
        return sorted(set(fallback))


@st.cache_data
def get_item_data(item_name):
    if not item_name:
        return None
    try:
        url = f"https://pokeapi.co/api/v2/item/{item_name.lower().replace(' ', '-')}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        return {"name": payload["name"].capitalize(), "sprite": payload["sprites"]["default"]}
    except Exception:
        return None


@st.cache_data
def get_move_data(move_name):
    if not move_name:
        return None
    try:
        url = f"https://pokeapi.co/api/v2/move/{move_name.lower().replace(' ', '-').strip()}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        return {
            "name": format_name(payload["name"]),
            "api_name": payload["name"],
            "type": payload["type"]["name"],
            "power": payload["power"] or 0,
            "accuracy": payload["accuracy"],
            "pp": payload["pp"],
            "damage_class": payload["damage_class"]["name"],
            "target": payload["target"]["name"],
            "stat_changes": [
                {"name": entry["stat"]["name"], "change": entry["change"]}
                for entry in payload.get("stat_changes", [])
            ],
        }
    except Exception:
        return None


@st.cache_data
def get_pokemon_data(name_or_id, item_name="Life Orb"):
    """
    Fetch BASE Pokémon data from PokeAPI.

    Returns sprite URLs, types, base_stats, battle-state fields, item info,
    and role_info.  Intentionally does NOT contain moves or move_pool — use
    generate_moveset() separately so that changing moveset_mode never
    invalidates or re-triggers this cached fetch.

    Cache key: (name_or_id, item_name)

    The private field "_all_move_names" carries the raw list of API move-name
    strings so that generate_moveset() can build the pool without a second
    PokeAPI call.
    """
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()

        api_stats = {entry["stat"]["name"]: entry["base_stat"] for entry in payload["stats"]}
        animated  = payload["sprites"]["versions"]["generation-v"]["black-white"]["animated"]
        img_front = animated["front_default"] or payload["sprites"]["front_default"]
        img_back  = animated["back_default"]  or payload["sprites"]["back_default"]

        base_stats = {
            "hp":     api_stats.get("hp", 0),
            "atk":    api_stats.get("attack", 0),
            "def":    api_stats.get("defense", 0),
            "sp_atk": api_stats.get("special-attack", 0),
            "sp_def": api_stats.get("special-defense", 0),
            "spd":    api_stats.get("speed", 0),
        }
        pokemon_types = [entry["type"]["name"] for entry in payload["types"]]

        # Collect all move names in API order — consumed by generate_moveset().
        all_move_names = [e["move"]["name"] for e in payload.get("moves", [])]

        return {
            "name":            payload["name"].capitalize(),
            "sprite_front":    img_front,
            "sprite_back":     img_back,
            "types":           pokemon_types,
            "base_stats":      base_stats,
            "stats":           dict(base_stats),
            "stat_stages":     {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0},
            "current_hp":      1.0,
            "status":          None,
            "item":            get_item_data(item_name),
            "debilitado":      False,
            "role_info":       get_role_info(base_stats),
            # Private — only consumed by generate_moveset(); callers should treat
            # this as an implementation detail and not rely on its contents.
            "_all_move_names": all_move_names,
        }
    except Exception:
        return None


def generate_moveset(base_data: dict, moveset_mode: str) -> tuple[list, list]:
    """
    Build a moveset for a Pokémon from its base data dict.

    This function is deliberately NOT cached at the composite level — the
    individual get_move_data() calls it makes are already cached, so repeated
    invocations with the same arguments are cheap.

    Returns (moves, move_pool) where:
        moves     — list of exactly 4 move dicts for the active battle slot
        move_pool — list of up to 20 scored candidate moves for Custom editing

    STRICT ISOLATION: This function never calls get_pokemon_data().
    Changing moveset_mode triggers no PokeAPI Pokémon fetches whatsoever.
    """
    all_move_names = base_data.get("_all_move_names", [])
    pokemon_types  = base_data["types"]
    base_stats     = base_data["base_stats"]
    pokemon_name   = base_data.get("name", "").lower()

    _struggle = {
        "name": "Struggle", "api_name": "struggle",
        "type": "normal", "power": 50, "accuracy": None,
        "pp": 1, "damage_class": "physical",
        "target": "selected-pokemon", "stat_changes": [],
    }

    # ── "random" mode: first 4 fetchable moves in PokeAPI order ──────────────
    if moveset_mode == "random":
        raw_moves: list[dict] = []
        for mn in all_move_names:
            md = get_move_data(mn)
            if md:
                raw_moves.append(md)
            if len(raw_moves) == 4:
                break
        while len(raw_moves) < 4:
            raw_moves.append(dict(_struggle))
        return raw_moves, raw_moves

    # ── competitive / balanced / custom: use the intelligent pipeline ─────────
    # Step 1: heuristic prefilter on name alone (no API calls)
    candidate_names = prefilter_move_names(all_move_names, pokemon_types, limit=22)

    # Step 2: fetch move details (each call is cached by get_move_data)
    candidates: list[dict] = []
    for mn in candidate_names:
        md = get_move_data(mn)
        if md:
            candidates.append(md)

    if not candidates:
        return [dict(_struggle)] * 4, []

    # Step 3: select final moveset
    # "custom" uses the same algorithm as "competitive" but also exposes the
    # full pool so the UI can render edit dropdowns.
    build_mode = "competitive" if moveset_mode == "custom" else moveset_mode
    moves = build_moveset(pokemon_name, pokemon_types, base_stats, candidates, mode=build_mode)

    # Step 4: build the pool for custom editing
    move_pool = get_filtered_move_pool(candidates, pokemon_types, base_stats, limit=20)

    return moves, move_pool


def reset_pokemon_state(pokemon):
    pokemon["current_hp"] = 1.0
    pokemon["status"] = None
    pokemon["debilitado"] = False
    pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    pokemon["stats"] = apply_stat_stages(pokemon["base_stats"], pokemon["stat_stages"])


def sync_env_with_active_pokemon():
    st.session_state.env.configure_battle(
        st.session_state.team_ia[st.session_state.active_ia],
        st.session_state.team_rival[st.session_state.active_rival],
    )


# Known status / utility move effect descriptions used in tooltips.
_KNOWN_MOVE_EFFECTS: dict[str, str] = {
    # Stat boosts — self
    "swords-dance":     "Raises user's Attack by +2 stages.",
    "nasty-plot":       "Raises user's Sp. Atk by +2 stages.",
    "calm-mind":        "Raises user's Sp. Atk and Sp. Def by +1 stage each.",
    "bulk-up":          "Raises user's Attack and Defense by +1 stage each.",
    "dragon-dance":     "Raises user's Attack and Speed by +1 stage each.",
    "quiver-dance":     "Raises user's Sp. Atk, Sp. Def, and Speed by +1 stage each.",
    "shell-smash":      "Drops Defense and Sp. Def by -1; raises Attack, Sp. Atk, Speed by +2.",
    "growth":           "Raises Sp. Atk by +1 (or +2 in harsh sunlight).",
    "iron-defense":     "Raises user's Defense by +2 stages.",
    "amnesia":          "Raises user's Sp. Def by +2 stages.",
    "hone-claws":       "Raises user's Attack and Accuracy by +1 stage each.",
    "work-up":          "Raises user's Attack and Sp. Atk by +1 stage each.",
    "coil":             "Raises user's Attack, Defense, and Accuracy by +1 stage each.",
    # Stat drops — opponent
    "growl":            "Lowers the opponent's Attack by -1 stage.",
    "tail-whip":        "Lowers the opponent's Defense by -1 stage.",
    "leer":             "Lowers the opponent's Defense by -1 stage.",
    "screech":          "Lowers the opponent's Defense by -2 stages.",
    "fake-tears":       "Lowers the opponent's Sp. Def by -2 stages.",
    "metal-sound":      "Lowers the opponent's Sp. Def by -2 stages.",
    "charm":            "Lowers the opponent's Attack by -2 stages.",
    "sweet-scent":      "Lowers the opponent's Evasiveness by -1 stage.",
    "string-shot":      "Lowers the opponent's Speed by -1 stage.",
    "scary-face":       "Lowers the opponent's Speed by -2 stages.",
    # Status conditions
    "thunder-wave":     "Paralyzes the opponent. Reduces Speed by 50%.",
    "toxic":            "Badly poisons the opponent (damage worsens each turn).",
    "poison-powder":    "Poisons the opponent.",
    "sleep-powder":     "Puts the opponent to sleep.",
    "hypnosis":         "Puts the opponent to sleep (60% accuracy).",
    "spore":            "Puts the opponent to sleep (100% accuracy).",
    "stun-spore":       "Paralyzes the opponent.",
    "will-o-wisp":      "Burns the opponent, halving its Attack.",
    "glare":            "Paralyzes the opponent.",
    "sing":             "Puts the opponent to sleep (55% accuracy).",
    # Recovery / healing
    "recover":          "Restores up to 50% of user's max HP.",
    "softboiled":       "Restores up to 50% of user's max HP.",
    "roost":            "Restores up to 50% of user's max HP. Removes Flying type this turn.",
    "slack-off":        "Restores up to 50% of user's max HP.",
    "milk-drink":       "Restores up to 50% of user's max HP.",
    "synthesis":        "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "moonlight":        "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "morning-sun":      "Restores HP (50% normally, 75% in sun, 25% in other weather).",
    "rest":             "User sleeps for 2 turns and fully restores HP + cures status.",
    "wish":             "At end of next turn, the Pokémon on field restores 50% of max HP.",
    "aqua-ring":        "Surrounds user in water to restore a little HP each turn.",
    "ingrain":          "Roots user in ground; restores HP each turn but prevents switching.",
    # Hazards / field
    "stealth-rock":     "Lays rocks that deal type-based damage (6–50%) to switching-in Pokémon.",
    "spikes":           "Lays spikes that damage (non-Flying) Pokémon when they switch in.",
    "toxic-spikes":     "Poisons Pokémon that switch in (2 layers = badly poisoned).",
    "sticky-web":       "Lowers Speed of grounded Pokémon that switch in.",
    "defog":            "Clears hazards and screens for both sides.",
    "rapid-spin":       "Clears entry hazards from user's side. Power 50.",
    # Protection / utility
    "protect":          "User is protected from most moves this turn. Fails if used consecutively.",
    "detect":           "Same as Protect (uses a separate turn counter).",
    "endure":           "User always survives this turn with at least 1 HP.",
    "substitute":       "Creates a decoy using 25% of max HP.",
    "encore":           "Forces opponent to repeat its last move for 3 turns.",
    "taunt":            "Prevents opponent from using status moves for 3 turns.",
    "trick":            "Swaps held items with the opponent.",
    "switcheroo":       "Swaps held items with the opponent.",
    "knock-off":        "Removes opponent's held item; deals bonus damage if item is present.",
    "leech-seed":       "Seeds the opponent; drains 1/8 of its HP each turn to heal the user.",
    "rain-dance":       "Summons rain for 5 turns. Boosts Water moves, weakens Fire moves.",
    "sunny-day":        "Summons harsh sunlight for 5 turns. Boosts Fire moves, weakens Water.",
    "sandstorm":        "Summons a sandstorm for 5 turns. Damages non-Rock/Ground/Steel types.",
    "hail":             "Summons hail for 5 turns. Damages non-Ice types.",
    "trick-room":       "Slower Pokémon move first for 5 turns.",
    "reflect":          "Halves physical damage taken by user's side for 5 turns.",
    "light-screen":     "Halves special damage taken by user's side for 5 turns.",
    "aurora-veil":      "Requires hail; halves both physical and special damage for 5 turns.",
    "baton-pass":       "Switches out, passing any stat changes to the next Pokémon.",
    "u-turn":           "Deals damage then switches the user out. Power 70.",
    "volt-switch":      "Deals damage then switches the user out. Power 70.",
}


def _describe_status_move(move: dict) -> str:
    """Return an effect description for status / utility moves."""
    name = (move.get("name") or "").lower().strip()
    return _KNOWN_MOVE_EFFECTS.get(name, "")


def get_move_tooltip(move: dict, defender: dict) -> str:
    """Rich tooltip for a move button — includes effect descriptions for status moves."""
    type_name    = format_name(move.get("type"))
    damage_class = format_name(move.get("damage_class"))
    power        = move.get("power") or 0
    accuracy     = move.get("accuracy") or 100
    multiplier   = get_type_multiplier(move.get("type"), defender.get("types", []))
    eff_label    = describe_effectiveness(multiplier)

    parts = [
        f"Type: {type_name}",
        f"Class: {damage_class}",
        f"Power: {power or '—'}",
        f"Accuracy: {accuracy}%",
        f"Effectiveness: {eff_label}",
    ]

    # Append effect description for status/utility moves
    if not power:
        effect = _describe_status_move(move)
        if effect:
            parts.append(f"Effect: {effect}")

    return " | ".join(parts)


def find_next_available(team):
    return next((idx for idx, pokemon in enumerate(team) if not pokemon["debilitado"]), None)


def _send_in_pokemon(side: str, new_idx: int):
    """Swap the active Pokémon for a side, applying hazard entry damage."""
    env = st.session_state.env
    team = st.session_state.team_ia if side == "ia" else st.session_state.team_rival
    hazards = env.hazards_ia if side == "ia" else env.hazards_rival
    pokemon = team[new_idx]
    pokemon["current_hp"] = max(0.0, float(pokemon.get("current_hp", 1.0)))
    # Apply entry hazard damage if any hazards are set
    if hazards:
        chip, haz_log = get_hazard_entry_damage(pokemon, hazards)
        if chip > 0:
            pokemon["current_hp"] = max(0.0, pokemon["current_hp"] - chip)
            st.session_state.historial.insert(0, f"📌 {haz_log}")
    if side == "ia":
        st.session_state.active_ia = new_idx
    else:
        st.session_state.active_rival = new_idx
    sync_env_with_active_pokemon()


def handle_post_turn_state():
    challenge_mode = st.session_state.get("battle_mode", "1. Simulación") == "2. Desafío"

    # ── Rival Pokémon fainted ────────────────────────────────────────────────
    if st.session_state.env.hp_rival <= 0:
        st.session_state.team_rival[st.session_state.active_rival]["debilitado"] = True
        next_rival = find_next_available(st.session_state.team_rival)
        if next_rival is not None:
            if challenge_mode:
                # In challenge mode the player must choose — set flag and wait
                st.session_state.must_switch_rival = True
            else:
                # Simulation: auto-advance to next available
                _send_in_pokemon("rival", next_rival)
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "🏆 ¡VICTORIA DE LA IA!"

    # ── IA Pokémon fainted ───────────────────────────────────────────────────
    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]["debilitado"] = True
        next_ia = find_next_available(st.session_state.team_ia)
        if next_ia is not None:
            _send_in_pokemon("ia", next_ia)
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "💀 LA IA HA SIDO DERROTADA"


if "game_started" not in st.session_state:
    st.session_state.update(
        {
            "game_started": False,
            "battle_finished": False,
            "resultado": "",
            "active_ia": 0,
            "active_rival": 0,
            "historial": [],
            # BattleEngine is the Game Logic Layer simulator.
            # It has the same interface as PokemonEnv's live-battle mode
            # but adds full battle mechanics (status, weather, hazards).
            # PokemonEnv is used ONLY in training scripts — never here.
            "env": BattleEngine(),
            "loaded_model": None,
            "current_model_path": "",
            "auto_enabled": False,
            # Turn counter shown in battle log
            "turn_number": 0,
            # When True in challenge mode, player must pick their next Pokémon
            "must_switch_rival": False,
            # Tracks the selected battle mode ("1. Simulación" / "2. Desafío")
            "battle_mode": "1. Simulación",
            # Moveset strategy selector ("competitive" / "balanced" / "random" / "custom")
            "moveset_mode": "competitive",
            # Per-Pokémon custom move overrides: {f"{prefix}_{idx}": [move_dict, …]}
            "custom_moves": {},
        }
    )


def predict_action_compatible(model, env):
    """
    Get the AI action for the current battle state.

    The *env* argument is now a BattleEngine instance (Game Logic Layer).
    BattleEngine._get_obs() delegates to obs_builder.build_obs_28(), which
    produces the IDENTICAL 28-dim vector that PokemonEnv._get_obs() would
    produce — so PPO models trained on PokemonEnv remain fully compatible.

    Decision pipeline
    -----------------
    1. If no model: pure greedy fallback (always valid, never immune moves).
    2. If model loaded: PPO.predict(obs) → raw_action.
    3. ai_advisor.get_hybrid_action() post-filters the raw action:
       - Blocks 0× (immune) move choices.
       - Overrides redundant status moves.
       - Prefers super-effective moves when they are clearly dominant.
       This filter is UI-only and does NOT affect PPO training.
    """
    ia_pokemon    = env.ia_pokemon
    rival_pokemon = env.rival_pokemon

    if model is None:
        if ia_pokemon and rival_pokemon:
            return get_greedy_action(ia_pokemon, rival_pokemon)
        raise RuntimeError("No PPO model loaded and no active Pokémon for greedy fallback.")

    # BattleEngine._get_obs() → obs_builder.build_obs_28() → 28-dim float32
    obs        = env._get_obs()
    ppo_action = int(model.predict(obs, deterministic=True)[0])

    if ia_pokemon and rival_pokemon:
        return get_hybrid_action(ppo_action, ia_pokemon, rival_pokemon)
    return ppo_action


@st.cache_data(show_spinner=False)
def get_compatible_model_catalog(models_dir):
    compatible = []
    incompatible = []
    for root, _, files in os.walk(models_dir):
        for file_name in files:
            if not file_name.endswith(".zip"):
                continue
            relative_zip = os.path.relpath(os.path.join(root, file_name), models_dir)
            model_base_rel = relative_zip[:-4]
            model_base = os.path.join(models_dir, model_base_rel)
            compat = check_model_compatibility(model_base)
            if compat.is_valid:
                compatible.append(relative_zip)
            else:
                incompatible.append((relative_zip, compat.reason))
    return sorted(compatible), sorted(incompatible)


def combat_step(action_ia, action_rival=None):
    sync_env_with_active_pokemon()
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    curr_rival = st.session_state.team_rival[st.session_state.active_rival]
    old_hp_rival, old_hp_ia = st.session_state.env.hp_rival, st.session_state.env.hp_ia

    _, _, _, _, info = st.session_state.env.step(action_ia, action_rival=action_rival)

    # Increment the global turn counter (env's turn_count resets on configure_battle
    # in challenge mode, so we maintain our own display counter here)
    st.session_state.turn_number = st.session_state.get("turn_number", 0) + 1
    turn_label = f"**T{st.session_state.turn_number}**"

    damage_to_rival = max(0, (old_hp_rival - st.session_state.env.hp_rival) * 100)
    damage_to_ia = max(0, (old_hp_ia - st.session_state.env.hp_ia) * 100)

    ia_eff = info.get("ia_effectiveness", "")
    rival_eff = info.get("rival_effectiveness", "")
    eff_tag_ia = f" `{ia_eff}`" if ia_eff and ia_eff not in ("Neutral", "") else ""
    eff_tag_rival = f" `{rival_eff}`" if rival_eff and rival_eff not in ("Neutral", "") else ""

    st.session_state.historial.insert(
        0,
        f"{turn_label} 🔴 **{curr_rival['name']}**: −{damage_to_ia:.1f}% HP | {info['rival_move']}{eff_tag_rival}",
    )
    st.session_state.historial.insert(
        0,
        f"{turn_label} ⚔️ **{curr_ia['name']}** (IA): −{damage_to_rival:.1f}% HP | {info['ia_move']}{eff_tag_ia}",
    )
    handle_post_turn_state()


def switch_rival_pokemon(new_index):
    if new_index == st.session_state.active_rival:
        return
    if st.session_state.team_rival[new_index]["debilitado"]:
        return

    action_ia = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
    old_active = st.session_state.active_rival
    sync_env_with_active_pokemon()
    _, _, _, _, info = st.session_state.env.switch_turn(
        side="rival",
        new_active_pokemon=st.session_state.team_rival[new_index],
        opponent_action=action_ia,
    )
    st.session_state.active_rival = new_index
    damage_to_rival = max(0, info.get("hp_change_rival", 0.0) * 100)
    if info.get("switch_log"):
        st.session_state.historial.insert(0, f"🔁 **{st.session_state.team_rival[new_index]['name']}**: {info['switch_log']}")
    if info.get("ia_move") and info["ia_move"] != "No attack":
        st.session_state.historial.insert(0, f"⚔️ **{st.session_state.team_ia[st.session_state.active_ia]['name']}** (IA): -{damage_to_rival:.1f}% | {info['ia_move']}")
    st.session_state.team_rival[old_active]["current_hp"] = st.session_state.team_rival[old_active].get("current_hp", 1.0)
    handle_post_turn_state()


def get_switch_options(team, active_index):
    options = []
    for idx, pokemon in enumerate(team):
        if idx == active_index or pokemon["debilitado"]:
            continue
        options.append((idx, f"{pokemon['name']} ({int(pokemon['current_hp'] * 100)}% HP)"))
    return options


if not st.session_state.game_started:
    st.title("🧪 Configuración de Equipos - TFM AI")
    pokemon_catalog = get_pokemon_catalog()
    item_catalog = get_item_catalog()
    st.caption("Los selectores son buscables: escribe para filtrar, navega con flechas y confirma con Enter.")

    # ── Moveset strategy selector ──────────────────────────────────────────
    st.divider()
    st.subheader("🧠 Estrategia de Moveset")

    _MODE_LABELS = {
        "🏆 Competitive (Auto)": "competitive",
        "⚖️ Balanced":           "balanced",
        "🎲 Random (Legacy)":    "random",
        "✏️ Custom":             "custom",
    }
    _MODE_HELP = {
        "🏆 Competitive (Auto)": "Selección inteligente: STAB + cobertura + utilidad según el rol del Pokémon.",
        "⚖️ Balanced":           "Como Competitive pero permite hasta 2 movimientos del mismo tipo.",
        "🎲 Random (Legacy)":    "Los primeros 4 movimientos de la PokeAPI (comportamiento original).",
        "✏️ Custom":             "Elige manualmente desde el pool filtrado competitivo de cada Pokémon.",
    }
    _mode_col1, _mode_col2 = st.columns([2, 3])
    with _mode_col1:
        _selected_label = st.radio(
            "Modo de moveset:",
            list(_MODE_LABELS.keys()),
            key="moveset_mode_radio",
            help="Afecta a ambos equipos. Puedes cambiar el modo antes de iniciar la batalla.",
        )
    with _mode_col2:
        st.info(_MODE_HELP[_selected_label])

    moveset_mode: str = _MODE_LABELS[_selected_label]
    st.session_state.moveset_mode = moveset_mode

    # ── Moveset-mode change detection ─────────────────────────────────────────
    # get_pokemon_data(name, item) has NO moveset_mode in its cache key, so
    # switching moveset strategy NEVER re-fetches the Pokémon payload from the
    # PokeAPI.  Moves are built by generate_moveset() which reuses individually-
    # cached get_move_data() calls — also zero extra network traffic on a hit.
    #
    # What we DO clear on a mode switch:
    #   • custom_moves   — manual overrides must not bleed into other modes
    #   • ia/riv_slot_*  — Custom-expander selectbox values stay in session_state
    #                       even when the expander is collapsed; stale values here
    #                       would silently override the freshly computed moves
    # We do NOT touch ia_n_* / riv_n_* — the Pokémon roster is independent.
    _prev_moveset_mode = st.session_state.get("_prev_moveset_mode")
    if _prev_moveset_mode != moveset_mode:
        st.session_state.pop("custom_moves", None)
        for _mi in range(6):
            for _mpfx in ("ia", "riv"):
                for _slot in range(4):
                    st.session_state.pop(f"{_mpfx}_slot_{_mi}_{_slot}", None)
        st.session_state["_prev_moveset_mode"] = moveset_mode

    st.divider()

    # ── Form post-processing ───────────────────────────────────────────────────
    # Applied AFTER both team selection (which Pokémon) and moveset resolution
    # (which moves) are complete.  Always returns a NEW dict — never mutates the
    # input — so the @st.cache_data result is never dirtied.
    #
    # Architecture:
    #   Phase 1  get_pokemon_base_data  →  sprite / types / base_stats
    #   Phase 2  get_pokemon_data       →  adds moves for moveset_mode
    #   Phase 3  _apply_form_transforms →  overlays Mega / Gmax / Dynamax changes
    #
    # Forms are purely cosmetic/stat overlays on the display dict.  They never
    # touch the RL observation (obs_builder) or the battle engine's base stats.

    # Known Mega stone → type overrides.  Stat multipliers are approximated by
    # scaling base_stats; exact values can be refined per Pokémon as needed.
    _MEGA_STONE_MAP: dict[str, dict] = {
        # item name (lowercase, spaces ok) → {types?, stat_mult?, name_suffix?}
        "charizardite x":  {"types": ["fire", "dragon"],  "stat_mult": 1.10, "suffix": "-Mega-X"},
        "charizardite y":  {"types": ["fire", "flying"],  "stat_mult": 1.10, "suffix": "-Mega-Y"},
        "blastoisinite":   {"types": ["water"],            "stat_mult": 1.10, "suffix": "-Mega"},
        "venusaurite":     {"types": ["grass", "poison"],  "stat_mult": 1.10, "suffix": "-Mega"},
        "gengarite":       {"types": ["ghost", "poison"],  "stat_mult": 1.10, "suffix": "-Mega"},
        "gardevoirite":    {"types": ["psychic", "fairy"], "stat_mult": 1.10, "suffix": "-Mega"},
        "lucarionite":     {"types": ["fighting", "steel"],"stat_mult": 1.10, "suffix": "-Mega"},
        "salamencite":     {"types": ["dragon", "flying"], "stat_mult": 1.10, "suffix": "-Mega"},
        "metagrossite":    {"types": ["steel", "psychic"], "stat_mult": 1.10, "suffix": "-Mega"},
        "rayquazite":      {"types": ["dragon", "flying"], "stat_mult": 1.10, "suffix": "-Mega"},
        "tyranitarite":    {"types": ["rock", "dark"],     "stat_mult": 1.10, "suffix": "-Mega"},
        "mewtwonite x":    {"types": ["psychic", "fighting"],"stat_mult": 1.12,"suffix": "-Mega-X"},
        "mewtwonite y":    {"types": ["psychic"],          "stat_mult": 1.12, "suffix": "-Mega-Y"},
    }

    def _apply_form_transforms(data: dict, item_name: str) -> dict:
        """
        Return a new Pokémon dict with form-based overlays applied.
        Input dict is NEVER mutated.

        Supported transforms (applied in order):
          1. Mega Evolution  — detected via item name in _MEGA_STONE_MAP
          2. Gigantamax      — detected via item name == "Gigantamax Factor" (placeholder)
          3. Dynamax         — detected via item name == "Dynamax Band" (placeholder)
        """
        item_key = (item_name or "").lower().strip()

        # ── Mega Evolution ────────────────────────────────────────────────────
        if item_key in _MEGA_STONE_MAP:
            spec = _MEGA_STONE_MAP[item_key]
            new_types    = spec.get("types", data["types"])
            stat_mult    = spec.get("stat_mult", 1.0)
            suffix       = spec.get("suffix", "-Mega")
            new_stats    = {k: int(v * stat_mult) for k, v in data["base_stats"].items()}
            return {
                **data,                               # shallow copy of all keys
                "name":       data["name"] + suffix,
                "types":      new_types,
                "base_stats": new_stats,
                "stats":      new_stats,              # display stats (not battle-engine)
                "_form":      "mega",
            }

        # ── Gigantamax (cosmetic placeholder — no stat changes yet) ───────────
        if item_key == "gigantamax factor":
            return {
                **data,
                "name":  data["name"] + "-Gmax",
                "_form": "gmax",
            }

        # ── Dynamax: HP doubled for display, mark for UI badge ────────────────
        if item_key == "dynamax band":
            new_stats = {
                **data["base_stats"],
                "hp": data["base_stats"].get("hp", 0) * 2,
            }
            return {
                **data,
                "base_stats": new_stats,
                "stats":      new_stats,
                "_form":      "dynamax",
            }

        # No form applicable — return a shallow copy so caller always owns it
        return {**data}

    # ── Team rendering ─────────────────────────────────────────────────────
    def render_team_selection(title: str, defaults: list, key_prefix: str):
        st.subheader(title)
        team = []
        cols = st.columns(3)

        for idx in range(6):
            with cols[idx % 3]:
                default_pokemon = (
                    defaults[idx] if defaults[idx] in pokemon_catalog else pokemon_catalog[0]
                )
                default_item = "Life Orb" if "Life Orb" in item_catalog else item_catalog[0]

                name = st.selectbox(
                    f"Pokémon {idx + 1}",
                    options=pokemon_catalog,
                    index=pokemon_catalog.index(default_pokemon),
                    key=f"{key_prefix}_n_{idx}",
                    placeholder="Escribe para buscar Pokémon",
                )
                item = st.selectbox(
                    f"Objeto {idx + 1}",
                    options=item_catalog,
                    index=item_catalog.index(default_item),
                    key=f"{key_prefix}_i_{idx}",
                    placeholder="Escribe para buscar objeto",
                )

                # ── Phase 1: base Pokémon data (cached per name+item only) ────
                # IMPORTANT: always materialise a fresh shallow copy before use.
                # @st.cache_data returns the same Python object on a cache hit
                # within a session.  If we kept a naked reference and later
                # mutated it (e.g. reset_pokemon_state), the cache entry would
                # be dirtied and every future hit would return corrupted data.
                # STRICT: get_pokemon_data has NO moveset_mode parameter — it
                # never re-fetches when the user switches moveset strategy.
                _cached = get_pokemon_data(name, item)
                if not _cached:
                    continue
                base_data = {**_cached}   # owned shallow copy — safe to modify

                # ── Phase 2: moveset generation (independent of base cache) ───
                # generate_moveset() builds moves from base_data["_all_move_names"]
                # using only individually-cached get_move_data() calls.
                # Changing moveset_mode costs zero extra PokeAPI Pokémon fetches.
                moves, move_pool = generate_moveset(base_data, moveset_mode)

                # Apply custom move overrides (only in custom mode)
                custom_key = f"{key_prefix}_{idx}"
                if moveset_mode == "custom":
                    override = st.session_state.get("custom_moves", {}).get(custom_key)
                    if override and len(override) == 4:
                        moves = override

                data = {**base_data, "moves": moves, "move_pool": move_pool}

                # ── Phase 3: form post-processing ─────────────────────────────
                # Applied after team resolution (which Pokémon) AND moveset
                # resolution (which moves).  Returns a new dict; never mutates.
                data = _apply_form_transforms(data, item)

                with st.container(border=True):
                    # Sprite + item
                    sprite_col, info_col = st.columns([1, 2])
                    sprite_col.image(data["sprite_front"], width=68)
                    with info_col:
                        st.markdown(f"**{data['name']}**")
                        # Type badges
                        badges_html = "".join(
                            type_badge_html(t) for t in data["types"]
                        )
                        st.markdown(badges_html, unsafe_allow_html=True)
                        # Role badge
                        ri = data.get("role_info", {})
                        if ri:
                            role_html = (
                                f'<span style="font-size:10px;font-weight:bold;'
                                f'color:{ri.get("color","#888")};">'
                                f'{ri.get("label","")}</span>'
                            )
                            st.markdown(role_html, unsafe_allow_html=True)
                        if data.get("item"):
                            st.image(data["item"]["sprite"], width=28,
                                     caption=data["item"]["name"])

                    st.divider()
                    # Move list with type badge + power
                    for move in data["moves"]:
                        mtype   = move.get("type", "normal")
                        mpower  = move.get("power") or 0
                        mdc     = move.get("damage_class", "status")
                        badge   = type_badge_html(mtype, small=True)
                        pwr_str = f"Pwr {mpower}" if mpower else "Status"
                        cls_icon = "⚔️" if mdc == "physical" else ("✨" if mdc == "special" else "🔮")
                        st.markdown(
                            f"{badge} {cls_icon} **{move['name']}** — {pwr_str}",
                            unsafe_allow_html=True,
                        )

                    # ── Custom editing expander ────────────────────────────
                    if moveset_mode == "custom" and base_data.get("move_pool"):
                        pool = base_data["move_pool"]
                        option_names  = [m["name"] for m in pool]
                        option_by_name = {m["name"]: m for m in pool}

                        with st.expander("✏️ Personalizar movimientos"):
                            current_moves = data["moves"]
                            new_custom: list[dict] = []

                            for slot in range(4):
                                default_name = (
                                    current_moves[slot]["name"]
                                    if slot < len(current_moves)
                                    else option_names[0]
                                )
                                safe_idx = (
                                    option_names.index(default_name)
                                    if default_name in option_names
                                    else 0
                                )
                                sel_name = st.selectbox(
                                    f"Slot {slot + 1}",
                                    options=option_names,
                                    index=safe_idx,
                                    key=f"{key_prefix}_slot_{idx}_{slot}",
                                )
                                new_custom.append(
                                    option_by_name.get(sel_name, pool[0])
                                )

                            # Persist selection in session state
                            if "custom_moves" not in st.session_state:
                                st.session_state.custom_moves = {}
                            st.session_state.custom_moves[custom_key] = new_custom
                            # Reflect changes in the card immediately
                            data = {**data, "moves": new_custom}

                team.append(data)
        return team

    # ── Team preset system ────────────────────────────────────────────────────
    # Curated competitive teams.  Each entry is (ia_team, rival_team) with 6
    # Pokémon names apiece.  Names must exist in the PokeAPI catalog.
    COMPETITIVE_PRESETS: dict[str, dict] = {
        "🏆 Balanced Offensive": {
            "description": "Versatile all-rounder team balancing physical, special, and utility.",
            "ia":    ["Mewtwo",    "Garchomp",  "Starmie",  "Scizor",    "Heatran",   "Togekiss"],
            "rival": ["Dragonite", "Tyranitar", "Gengar",   "Gyarados",  "Salamence", "Lucario"],
        },
        "🐉 Dragon Slayer": {
            "description": "Ice + Fairy + Steel coverage to dismantle Dragon-heavy teams.",
            "ia":    ["Mamoswine", "Clefable",  "Scizor",   "Weavile",   "Togekiss",  "Jolteon"],
            "rival": ["Rayquaza",  "Dragonite", "Garchomp", "Salamence", "Latios",    "Kingdra"],
        },
        "☀️ Sun Team": {
            "description": "Drought-powered team: Solar Beam, Fire moves boosted, Chlorophyll sweepers.",
            "ia":    ["Charizard", "Venusaur",  "Ninetales", "Heatran",  "Victreebel","Arcanine"],
            "rival": ["Kyogre",    "Blastoise", "Vaporeon",  "Politoed", "Ludicolo",  "Swampert"],
        },
        "🌊 Rain Squad": {
            "description": "Swift Swim sweepers backed by Drizzle rain support.",
            "ia":    ["Kyogre",    "Kabutops",  "Ludicolo",  "Toxicroak","Kingdra",   "Starmie"],
            "rival": ["Groudon",   "Charizard", "Ninetales", "Heatran",  "Volcarona", "Arcanine"],
        },
        "👻 Ghost & Psychic": {
            "description": "Trick Room + Psychic/Ghost types for mind-game dominance.",
            "ia":    ["Mewtwo",    "Gengar",    "Alakazam",  "Slowbro",  "Chandelure","Espeon"],
            "rival": ["Tyranitar", "Krookodile","Bisharp",   "Incineroar","Pangoro",  "Absol"],
        },
    }

    # ── Team selection mode ───────────────────────────────────────────────────
    st.divider()
    st.subheader("👥 Selección de Equipos")
    _team_mode_options = ["🎲 Random", "✏️ Custom", "🏆 Competitive Preset"]
    _team_mode = st.radio(
        "Modo de equipo:",
        _team_mode_options,
        index=0,
        key="team_selection_mode",
        horizontal=True,
        help=(
            "Random: equipos por defecto variados. "
            "Custom: elige tus propios Pokémon. "
            "Competitive Preset: equipos curados listos para competir."
        ),
    )

    # ── Helper: pick 6 unique random Pokémon from the catalog ─────────────────
    def _pick_random_team() -> list[str]:
        pool = pokemon_catalog if len(pokemon_catalog) >= 6 else (pokemon_catalog * 6)
        return random.sample(pool, 6)

    # ── STEP 1 ─ Render conditional widgets so their values are known ──────────
    # The preset selectbox must render (and store its value) BEFORE we do
    # mode-change detection, because the detection needs _cur_preset.
    _cur_preset = ""
    if _team_mode == "🏆 Competitive Preset":
        _preset_names = list(COMPETITIVE_PRESETS.keys())
        _chosen_preset_label = st.selectbox(
            "Elige un preset competitivo:",
            _preset_names,
            key="competitive_preset_choice",
        )
        _cur_preset = _chosen_preset_label

    # ── STEP 2 ─ Mode-change detection (runs with full info) ──────────────────
    # We now know both _team_mode and _cur_preset.  Only here can we correctly
    # decide whether anything changed and what to regenerate.
    #
    # Critical insight: this block must run BEFORE we set _default_ia/_default_rival.
    # If we set defaults first and detect changes second, any session_state writes
    # (e.g. rand_team_ia) made in step 2 would be invisible to the defaults that
    # were already assigned — the classic ordering race condition.
    _prev_mode   = st.session_state.get("_prev_team_mode")
    _prev_preset = st.session_state.get("_prev_preset_choice", "")

    _mode_changed   = _prev_mode != _team_mode
    _preset_changed = _team_mode == "🏆 Competitive Preset" and _prev_preset != _cur_preset

    if _mode_changed or _preset_changed:
        # Wipe all Pokémon-name and item selectbox keys so that the index= param
        # in render_team_selection() is honoured instead of the stale stored value.
        for _i in range(6):
            for _pfx in ("ia", "riv"):
                st.session_state.pop(f"{_pfx}_n_{_i}", None)
                st.session_state.pop(f"{_pfx}_i_{_i}", None)
        st.session_state.pop("custom_moves", None)

        # For Random: generate the new team NOW (selectbox keys were just cleared,
        # so render_team_selection will pick up these names via index=).
        # For other modes: discard any previously stored random team.
        if _team_mode == "🎲 Random":
            st.session_state["rand_team_ia"]    = _pick_random_team()
            st.session_state["rand_team_rival"] = _pick_random_team()
        else:
            st.session_state.pop("rand_team_ia",    None)
            st.session_state.pop("rand_team_rival", None)

        st.session_state["_prev_team_mode"]     = _team_mode
        st.session_state["_prev_preset_choice"] = _cur_preset

    # ── STEP 3 ─ Resolve defaults (reads session_state written in step 2) ──────
    _HARDCODED_IA    = ["Mewtwo",    "Rayquaza",  "Kyogre",   "Groudon",   "Metagross", "Sceptile"]
    _HARDCODED_RIVAL = ["Charizard", "Blastoise", "Venusaur", "Gengar",    "Lucario",   "Tyranitar"]

    _default_ia:    list[str] = _HARDCODED_IA
    _default_rival: list[str] = _HARDCODED_RIVAL

    if _team_mode == "🏆 Competitive Preset":
        _preset = COMPETITIVE_PRESETS[_cur_preset]
        st.caption(f"📋 {_preset['description']}")
        _default_ia    = _preset["ia"]
        _default_rival = _preset["rival"]

    elif _team_mode == "🎲 Random":
        # rand_team_ia was set in step 2 (on mode entry) or persists from the
        # previous render.  Either way it is always valid here.
        _default_ia    = st.session_state.get("rand_team_ia",    _HARDCODED_IA)
        _default_rival = st.session_state.get("rand_team_rival", _HARDCODED_RIVAL)

        _rc1, _rc2 = st.columns([3, 1])
        with _rc1:
            st.caption("Equipos generados aleatoriamente. Pulsa el botón para obtener nuevos Pokémon.")
        with _rc2:
            if st.button("🎲 Regenerar", key="btn_regen_random", use_container_width=True):
                # Generate new teams, wipe selectbox keys, then update _default_*
                # for THIS render so render_team_selection receives the new names.
                _new_ia    = _pick_random_team()
                _new_rival = _pick_random_team()
                st.session_state["rand_team_ia"]    = _new_ia
                st.session_state["rand_team_rival"] = _new_rival
                for _ri in range(6):
                    for _rpfx in ("ia", "riv"):
                        st.session_state.pop(f"{_rpfx}_n_{_ri}", None)
                        st.session_state.pop(f"{_rpfx}_i_{_ri}", None)
                st.session_state.pop("custom_moves", None)
                _default_ia    = _new_ia    # override for current render
                _default_rival = _new_rival

    elif _team_mode == "✏️ Custom":
        st.caption("Elige tus Pokémon manualmente. Los movimientos no se sobreescriben al cambiar estrategia.")

    col_ia, col_rival = st.columns(2)
    with col_ia:
        team_ia = render_team_selection(
            "🤖 Equipo IA",
            _default_ia,
            "ia",
        )
    with col_rival:
        team_rival = render_team_selection(
            "👤 Equipo Rival",
            _default_rival,
            "riv",
        )

    if st.button("🔥 INICIAR COMBATE", type="primary", use_container_width=True):
        if len(team_ia) == 6 and len(team_rival) == 6:
            try:
                conn = sqlite3.connect("pokemon_bigdata.db")
                conn.execute("DROP TABLE IF EXISTS v_logs")
                conn.execute(
                    """
                    CREATE TABLE v_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ia_move_name TEXT,
                        rival_move TEXT,
                        ia_move_type TEXT,
                        rival_move_type TEXT,
                        ia_effectiveness TEXT,
                        rival_effectiveness TEXT,
                        hp_ia REAL,
                        hp_rival REAL,
                        reward REAL
                    )
                    """
                )
                conn.close()
            except Exception:
                pass

            for pokemon in team_ia + team_rival:
                reset_pokemon_state(pokemon)

            st.session_state.update({"team_ia": team_ia, "team_rival": team_rival, "game_started": True})
            sync_env_with_active_pokemon()
            st.rerun()
        else:
            st.error("Asegúrate de que todos los Pokémon hayan cargado correctamente.")
    st.stop()


sync_env_with_active_pokemon()

with st.sidebar:
    st.title("🕹️ Panel de Control")
    mode = st.radio("Modo:", ["1. Simulación", "2. Desafío"], key="battle_mode")

    model_list, incompatible_models = get_compatible_model_catalog(MODELS_DIR)
    if model_list:
        selected_model = st.selectbox("Modelo PPO compatible:", model_list)
        model_path = os.path.join(MODELS_DIR, selected_model)
        if st.session_state.current_model_path != model_path:
            model_base = model_path[:-4]
            st.session_state.loaded_model = require_compatible_model(model_base)
            st.session_state.current_model_path = model_path
            st.success(f"Cerebro cargado: {selected_model}")
    else:
        st.error("No compatible PPO models were found. Train a new model with the canonical environment first.")
    if incompatible_models:
        with st.expander("Modelos bloqueados (LEGACY - INCOMPATIBLE)"):
            for model_name, reason in incompatible_models:
                st.caption(f"{model_name}: {reason}")

    auto = st.toggle("Auto-Play", value=st.session_state.auto_enabled, disabled=st.session_state.loaded_model is None)
    st.session_state.auto_enabled = auto
    speed = st.slider("Velocidad", 0.1, 2.0, 0.5)

    current_ia = st.session_state.team_ia[st.session_state.active_ia]
    st.divider()
    # Show active Pokémon role badge if available
    ri = current_ia.get("role_info", {})
    if ri:
        st.markdown(
            f'<span style="font-size:11px;font-weight:bold;color:{ri.get("color","#888")};">'
            f'{ri.get("label","")} — {ri.get("desc","")}</span>',
            unsafe_allow_html=True,
        )
    # Show moveset mode
    _mode_label = {
        "competitive": "🏆 Competitive", "balanced": "⚖️ Balanced",
        "random": "🎲 Random", "custom": "✏️ Custom",
    }.get(st.session_state.get("moveset_mode", "competitive"), "")
    if _mode_label:
        st.caption(f"Moveset: {_mode_label}")
    st.subheader(f"📊 Stats: {current_ia['name']}")
    st.table(pd.Series(current_ia["stats"]))
    st.caption(f"Tipos: {' / '.join(format_name(type_name) for type_name in current_ia['types'])}")
    st.caption(f"HP actual: {int(current_ia['current_hp'] * 100)}%")

    with st.expander("Tabla de tipos"):
        st.dataframe(pd.DataFrame(build_type_chart_rows()), use_container_width=True, hide_index=True)

if st.session_state.loaded_model is None:
    st.error("No compatible PPO model is loaded. Battle execution is blocked.")
    st.stop()


current_ia = st.session_state.team_ia[st.session_state.active_ia]
current_rival = st.session_state.team_rival[st.session_state.active_rival]

# Pre-compute values used in the arena HTML
_hp_ia    = float(st.session_state.env.hp_ia)
_hp_rival = float(st.session_state.env.hp_rival)
_bar_ia    = hp_bar_color(_hp_ia)
_bar_rival = hp_bar_color(_hp_rival)
_types_ia_html    = "".join(type_badge_html(t) for t in current_ia["types"])
_types_rival_html = "".join(type_badge_html(t) for t in current_rival["types"])
_status_ia_html    = status_badge_html(current_ia.get("status"))
_status_rival_html = status_badge_html(current_rival.get("status"))
_weather_html = weather_badge_html(st.session_state.env.weather)
_turn_html = (
    f'<div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);'
    f'background:rgba(0,0,0,0.7);color:#fff;padding:3px 12px;border-radius:8px;'
    f'font-size:13px;">Turno {st.session_state.get("turn_number", 0)}'
    f"{_weather_html}</div>"
    if st.session_state.get("turn_number", 0) > 0 else ""
)

st.html(
    f"""
    <div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png');
         background-size: cover; height: 320px; border-radius: 20px;
         position: relative; border: 3px solid #444;">
      {_turn_html}
      <!-- Rival (top-right) -->
      <div style="position:absolute;top:30px;right:50px;width:260px;
                  background:rgba(0,0,0,0.82);padding:10px 12px;
                  border-radius:12px;color:white;border-left:5px solid #ff4b4b;">
        <b style="font-size:15px;">{current_rival['name']}</b>
        {_status_rival_html}
        <div style="margin:3px 0;">{_types_rival_html}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
          <div style="flex:1;background:#333;height:10px;border-radius:5px;">
            <div style="width:{_hp_rival*100:.1f}%;background:{_bar_rival};
                        height:100%;border-radius:5px;transition:width 0.3s;"></div>
          </div>
          <span style="font-size:12px;min-width:36px;text-align:right;">
            {int(_hp_rival*100)}%
          </span>
        </div>
        <img src="{current_rival['sprite_front']}"
             style="position:absolute;top:70px;right:10px;" width="96">
      </div>
      <!-- IA (bottom-left) -->
      <div style="position:absolute;bottom:30px;left:50px;width:260px;
                  background:rgba(0,0,0,0.82);padding:10px 12px;
                  border-radius:12px;color:white;border-left:5px solid #00d4ff;">
        <b style="font-size:15px;">{current_ia['name']} (IA)</b>
        {_status_ia_html}
        <div style="margin:3px 0;">{_types_ia_html}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
          <div style="flex:1;background:#333;height:10px;border-radius:5px;">
            <div style="width:{_hp_ia*100:.1f}%;background:{_bar_ia};
                        height:100%;border-radius:5px;transition:width 0.3s;"></div>
          </div>
          <span style="font-size:12px;min-width:36px;text-align:right;">
            {int(_hp_ia*100)}%
          </span>
        </div>
        <img src="{current_ia['sprite_back']}"
             style="position:absolute;bottom:55px;left:10px;" width="112">
      </div>
    </div>
    """
)

m_ia, m_rival = st.columns(2)
with m_ia:
    cols = st.columns(6)
    for idx, pokemon in enumerate(st.session_state.team_ia):
        cols[idx].markdown(
            f'<div style="text-align:center; opacity:{"1" if not pokemon["debilitado"] else "0.3"}; border: 2px solid {"#00d4ff" if idx == st.session_state.active_ia else "transparent"}; border-radius:10px;"><img src="{pokemon["sprite_front"]}" width="45"></div>',
            unsafe_allow_html=True,
        )
with m_rival:
    cols = st.columns(6)
    for idx, pokemon in enumerate(st.session_state.team_rival):
        cols[idx].markdown(
            f'<div style="text-align:center; opacity:{"1" if not pokemon["debilitado"] else "0.3"}; border: 2px solid {"#ff4b4b" if idx == st.session_state.active_rival else "transparent"}; border-radius:10px;"><img src="{pokemon["sprite_front"]}" width="45"></div>',
            unsafe_allow_html=True,
        )

st.divider()

col_stats, col_log, col_actions = st.columns([1, 1.2, 1])

with col_stats:
    st.subheader("📊 Comparativa")
    st.table(pd.DataFrame({"IA (Aliado)": current_ia["stats"], "Rival": current_rival["stats"]}))
    # Coloured type badges for both active Pokémon
    st.markdown(
        "**IA:** " + "".join(type_badge_html(t) for t in current_ia["types"]) +
        "<br>**Rival:** " + "".join(type_badge_html(t) for t in current_rival["types"]),
        unsafe_allow_html=True,
    )
    if current_ia.get("status"):
        st.markdown(f"IA status: {status_badge_html(current_ia['status'])}", unsafe_allow_html=True)
    if current_rival.get("status"):
        st.markdown(f"Rival status: {status_badge_html(current_rival['status'])}", unsafe_allow_html=True)

with col_log:
    st.subheader("📜 Registro de Combate")
    with st.container(height=320):
        for entry in st.session_state.historial:
            st.write(entry)

with col_actions:
    if st.session_state.battle_finished:
        st.success(st.session_state.resultado)

    # ── Post-faint forced switch (challenge mode only) ──────────────────────
    elif st.session_state.get("must_switch_rival") and mode == "2. Desafío":
        st.subheader("💀 ¡Tu Pokémon se ha debilitado!")
        st.write("Elige tu siguiente Pokémon:")
        for idx, pokemon in enumerate(st.session_state.team_rival):
            if pokemon["debilitado"]:
                continue
            hp_pct = int(pokemon.get("current_hp", 1.0) * 100)
            type_html = "".join(type_badge_html(t, small=True) for t in pokemon["types"])
            btn_label = f"➡️ {pokemon['name']}  ({hp_pct}% HP)"
            if st.button(btn_label, key=f"forcedswitch_{idx}", use_container_width=True):
                _send_in_pokemon("rival", idx)
                st.session_state.must_switch_rival = False
                st.rerun()
            st.markdown(type_html, unsafe_allow_html=True)

    elif mode == "2. Desafío":
        st.subheader("🕹️ Tus Ataques")

        # ── Shared move-button chrome (injected once per render) ───────────
        # The CSS uses the :has() pseudo-class (Chrome 105+, Firefox 121+,
        # Safari 15.4+) to target the st.button() element immediately after
        # each hidden marker div, applying the per-type background colour.
        # This makes the ENTIRE button surface clickable while keeping full
        # type-colour styling — no separate action button needed.
        st.markdown("""
        <style>
        /* Shared chrome for ALL move buttons in this section */
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button {
            border: 1px solid rgba(255,255,255,0.20) !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            text-align: left !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            white-space: pre-line !important;
            line-height: 1.55 !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.28) !important;
            min-height: 58px !important;
            width: 100% !important;
            transition: filter 0.14s ease, transform 0.10s ease !important;
        }
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button:hover {
            filter: brightness(1.14) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.35) !important;
        }
        div[data-testid="stMarkdownContainer"]:has([id^="mvmkr"]) \
+ div[data-testid="stButton"] button:active {
            filter: brightness(0.92) !important;
            transform: translateY(0px) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        for idx, move in enumerate(current_rival["moves"]):
            effectiveness = get_type_multiplier(move.get("type"), current_ia.get("types", []))
            eff_label     = describe_effectiveness(effectiveness)

            type_name  = (move.get("type") or "normal").lower().strip()
            colors     = get_type_colors(type_name)
            bg, fg     = colors["bg"], colors["text"]
            emoji      = get_type_emoji(type_name)
            type_label = type_name.capitalize()
            move_name  = (move.get("name") or "???").replace("-", " ").title()
            power      = move.get("power")
            pwr_str    = f"PWR {power}" if power else "Status"

            # Effectiveness suffix — only shown for non-neutral hits
            eff_map = {
                "Super effective":   "  ✦ Super effective!",
                "Not very effective": "  ▼ Not very effective",
                "Immune":            "  ✕ No effect",
            }
            eff_suffix = eff_map.get(eff_label, "")

            # ── Inject hidden marker + per-button colour override ──────────
            # The :has(#mvmkrN) rule below overrides ONLY this button's
            # bg/fg while the shared rule above handles shape, size, hover.
            st.markdown(
                f"""<style>
div[data-testid="stMarkdownContainer"]:has(#mvmkr{idx}) \
+ div[data-testid="stButton"] button {{
    background-color: {bg} !important;
    color: {fg} !important;
}}
</style>
<div id="mvmkr{idx}" style="display:none;height:0;overflow:hidden;"></div>""",
                unsafe_allow_html=True,
            )

            # ── The entire button IS the card ──────────────────────────────
            # Line 1: type emoji + type label (left)   move name (right-ish)
            # Line 2: power or status info             effectiveness note
            btn_label = (
                f"{emoji} {type_label:<10}  {move_name}\n"
                f"    {pwr_str}{eff_suffix}"
            )

            if st.button(
                btn_label,
                key=f"at_{idx}",
                use_container_width=True,
                help=get_move_tooltip(move, current_ia),
            ):
                ia_action = predict_action_compatible(
                    st.session_state.loaded_model, st.session_state.env
                )
                combat_step(ia_action, action_rival=idx)
                st.rerun()

        st.divider()
        st.subheader("🔁 Cambiar Pokémon")
        switch_options = get_switch_options(st.session_state.team_rival, st.session_state.active_rival)
        if not switch_options:
            st.caption("No hay otros Pokémon disponibles para cambiar.")
        else:
            for switch_idx, switch_label in switch_options:
                if st.button(switch_label, key=f"switch_{switch_idx}", use_container_width=True):
                    switch_rival_pokemon(switch_idx)
                    st.rerun()
    else:
        if not auto:
            st.warning("⏸️ Simulación pausada.")
        elif not st.session_state.battle_finished:
            ia_action = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
            combat_step(ia_action)
            time.sleep(speed)
            st.rerun()


if st.session_state.battle_finished:
    st.divider()
    st.header("📊 Informe Analítico Post-Combate")
    try:
        conn = sqlite3.connect("pokemon_bigdata.db")
        df_hp = pd.read_sql("SELECT id, hp_ia, hp_rival FROM v_logs ORDER BY id ASC", conn)
        if not df_hp.empty:
            st.subheader("📈 Evolución de Vitalidad")
            st.line_chart(df_hp.set_index("id"))

            col_ia_report, col_rival_report = st.columns(2)
            with col_ia_report:
                st.subheader("⚔️ Movimientos IA")
                df_ia = pd.read_sql(
                    """
                    SELECT ia_move_name AS Movimiento, ia_move_type AS Tipo, ia_effectiveness AS Efectividad, COUNT(*) AS Usos
                    FROM v_logs
                    GROUP BY ia_move_name, ia_move_type, ia_effectiveness
                    """,
                    conn,
                )
                st.dataframe(df_ia, use_container_width=True)
            with col_rival_report:
                st.subheader("🛡️ Movimientos Rival")
                df_rival = pd.read_sql(
                    """
                    SELECT rival_move AS Movimiento, rival_move_type AS Tipo, rival_effectiveness AS Efectividad, COUNT(*) AS Usos
                    FROM v_logs
                    GROUP BY rival_move, rival_move_type, rival_effectiveness
                    """,
                    conn,
                )
                st.dataframe(df_rival, use_container_width=True)
        conn.close()
    except Exception as exc:
        st.error(f"Error cargando informe: {exc}")

    if st.button("🔄 REINICIAR TODO", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()


with st.expander("📊 Explorador de Big Data (Dataset Completo)"):
    try:
        conn = sqlite3.connect("pokemon_bigdata.db")
        df_full = pd.read_sql("SELECT * FROM pokemon_stats", conn)
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Total Pokémon Ingeridos", len(df_full))
        if not df_full.empty:
            metric_col_2.metric("Tipo más común", df_full["type1"].mode()[0].capitalize())
            st.dataframe(df_full, use_container_width=True)
        else:
            st.warning("La tabla está vacía. Ejecuta el ETL primero.")
        conn.close()
    except Exception:
        st.info("Consejo: ejecuta `etl_process.py` para cargar los datos de la API en SQL y ver esta sección.")
