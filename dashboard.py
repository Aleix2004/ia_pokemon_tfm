import os
import random
import sqlite3
import time

import pandas as pd
import requests
import streamlit as st
from stable_baselines3 import PPO

from src.battle_utils import (
    apply_stat_stages,
    build_type_chart_rows,
    describe_effectiveness,
    format_name,
    get_type_multiplier,
)
from src.env.pokemon_env import PokemonEnv


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
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
        api_stats = {entry["stat"]["name"]: entry["base_stat"] for entry in payload["stats"]}
        animated = payload["sprites"]["versions"]["generation-v"]["black-white"]["animated"]
        img_front = animated["front_default"] or payload["sprites"]["front_default"]
        img_back = animated["back_default"] or payload["sprites"]["back_default"]

        base_stats = {
            "hp": api_stats.get("hp", 0),
            "atk": api_stats.get("attack", 0),
            "def": api_stats.get("defense", 0),
            "sp_atk": api_stats.get("special-attack", 0),
            "sp_def": api_stats.get("special-defense", 0),
            "spd": api_stats.get("speed", 0),
        }

        moves = []
        for move_entry in payload["moves"]:
            move_data = get_move_data(move_entry["move"]["name"])
            if move_data:
                moves.append(move_data)
            if len(moves) == 4:
                break

        if not moves:
            return None

        while len(moves) < 4:
            moves.append(dict(moves[-1]))

        return {
            "name": payload["name"].capitalize(),
            "sprite_front": img_front,
            "sprite_back": img_back,
            "types": [entry["type"]["name"] for entry in payload["types"]],
            "base_stats": base_stats,
            "stats": dict(base_stats),
            "stat_stages": {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0},
            "current_hp": 1.0,
            "item": get_item_data(item_name),
            "debilitado": False,
            "moves": moves,
        }
    except Exception:
        return None


def reset_pokemon_state(pokemon):
    pokemon["current_hp"] = 1.0
    pokemon["debilitado"] = False
    pokemon["stat_stages"] = {"atk": 0, "def": 0, "sp_atk": 0, "sp_def": 0, "spd": 0}
    pokemon["stats"] = apply_stat_stages(pokemon["base_stats"], pokemon["stat_stages"])


def sync_env_with_active_pokemon():
    st.session_state.env.configure_battle(
        st.session_state.team_ia[st.session_state.active_ia],
        st.session_state.team_rival[st.session_state.active_rival],
    )


def get_move_tooltip(move, defender):
    multiplier = get_type_multiplier(move.get("type"), defender.get("types", []))
    return (
        f"Type: {format_name(move.get('type'))} | "
        f"Class: {format_name(move.get('damage_class'))} | "
        f"Power: {move.get('power') or 0} | "
        f"Effectiveness: {describe_effectiveness(multiplier)}"
    )


def find_next_available(team):
    return next((idx for idx, pokemon in enumerate(team) if not pokemon["debilitado"]), None)


def handle_post_turn_state():
    if st.session_state.env.hp_rival <= 0:
        st.session_state.team_rival[st.session_state.active_rival]["debilitado"] = True
        next_rival = find_next_available(st.session_state.team_rival)
        if next_rival is not None:
            st.session_state.active_rival = next_rival
            st.session_state.team_rival[next_rival]["current_hp"] = 1.0
            sync_env_with_active_pokemon()
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "🏆 ¡VICTORIA DE LA IA!"

    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]["debilitado"] = True
        next_ia = find_next_available(st.session_state.team_ia)
        if next_ia is not None:
            st.session_state.active_ia = next_ia
            st.session_state.team_ia[next_ia]["current_hp"] = 1.0
            sync_env_with_active_pokemon()
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
            "env": PokemonEnv(),
            "loaded_model": None,
            "current_model_path": "",
            "auto_enabled": False,
        }
    )


def predict_action_compatible(model, env):
    obs = env._get_obs() if hasattr(env, "_get_obs") else env.reset()[0]
    try:
        return int(model.predict(obs)[0])
    except Exception:
        return random.randint(0, 3)


def combat_step(action_ia, action_rival=None):
    sync_env_with_active_pokemon()
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    curr_rival = st.session_state.team_rival[st.session_state.active_rival]
    old_hp_rival, old_hp_ia = st.session_state.env.hp_rival, st.session_state.env.hp_ia

    _, _, _, _, info = st.session_state.env.step(action_ia, action_rival=action_rival)

    damage_to_rival = max(0, (old_hp_rival - st.session_state.env.hp_rival) * 100)
    damage_to_ia = max(0, (old_hp_ia - st.session_state.env.hp_ia) * 100)

    st.session_state.historial.insert(0, f"🔴 **{curr_rival['name']}**: -{damage_to_ia:.1f}% | {info['rival_move']}")
    st.session_state.historial.insert(0, f"⚔️ **{curr_ia['name']}** (IA): -{damage_to_rival:.1f}% | {info['ia_move']}")
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

    def render_team_selection(title, defaults, key_prefix):
        st.subheader(title)
        team = []
        cols = st.columns(3)
        for idx in range(6):
            with cols[idx % 3]:
                default_pokemon = defaults[idx] if defaults[idx] in pokemon_catalog else pokemon_catalog[0]
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
                data = get_pokemon_data(name, item)
                if data:
                    with st.container(border=True):
                        sprite_col, item_col = st.columns([1, 1])
                        sprite_col.image(data["sprite_front"], width=70)
                        if data["item"]:
                            item_col.image(data["item"]["sprite"], width=40, caption=data["item"]["name"])
                        st.caption(f"**{data['name']}**")
                        st.caption(" / ".join(format_name(type_name) for type_name in data["types"]))
                        for move in data["moves"]:
                            st.caption(f"{move['name']} [{format_name(move['type'])}]")
                    team.append(data)
        return team

    col_ia, col_rival = st.columns(2)
    with col_ia:
        team_ia = render_team_selection(
            "🤖 Equipo IA",
            ["Mewtwo", "Rayquaza", "Kyogre", "Groudon", "Metagross", "Sceptile"],
            "ia",
        )
    with col_rival:
        team_rival = render_team_selection(
            "👤 Equipo Rival",
            ["Charizard", "Blastoise", "Venusaur", "Gengar", "Lucario", "Tyranitar"],
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
    mode = st.radio("Modo:", ["1. Simulación", "2. Desafío"])

    model_list = []
    for root, _, files in os.walk(MODELS_DIR):
        for file_name in files:
            if file_name.endswith(".zip"):
                model_list.append(os.path.relpath(os.path.join(root, file_name), MODELS_DIR))

    if model_list:
        selected_model = st.selectbox("Modelo PPO:", sorted(model_list))
        model_path = os.path.join(MODELS_DIR, selected_model)
        if st.session_state.current_model_path != model_path:
            try:
                st.session_state.loaded_model = PPO.load(model_path)
                st.session_state.current_model_path = model_path
                st.success(f"Cerebro cargado: {selected_model}")
            except Exception as exc:
                st.error(f"Error al cargar {selected_model}: {exc}")
    else:
        st.error(f"No se encontraron modelos .zip en {MODELS_DIR}")

    auto = st.toggle("Auto-Play", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto
    speed = st.slider("Velocidad", 0.1, 2.0, 0.5)

    current_ia = st.session_state.team_ia[st.session_state.active_ia]
    st.divider()
    st.subheader(f"📊 Stats: {current_ia['name']}")
    st.table(pd.Series(current_ia["stats"]))
    st.caption(f"Tipos: {' / '.join(format_name(type_name) for type_name in current_ia['types'])}")
    st.caption(f"HP actual: {int(current_ia['current_hp'] * 100)}%")

    with st.expander("Tabla de tipos"):
        st.dataframe(pd.DataFrame(build_type_chart_rows()), use_container_width=True, hide_index=True)


current_ia = st.session_state.team_ia[st.session_state.active_ia]
current_rival = st.session_state.team_rival[st.session_state.active_rival]

st.html(
    f"""
    <div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 300px; border-radius: 20px; position: relative; border: 3px solid #444;">
        <div style="position: absolute; top: 30px; right: 50px; width: 240px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 10px; color: white; border-left: 5px solid #ff4b4b;">
            <b>{current_rival['name']}</b>
            <div>{' / '.join(format_name(type_name) for type_name in current_rival['types'])}</div>
            <span style="float: right;">{int(st.session_state.env.hp_rival * 100)}%</span>
            <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; margin-top: 5px;"><div style="width: {st.session_state.env.hp_rival * 100}%; background: #4CAF50; height: 100%; border-radius: 5px;"></div></div>
            <img src="{current_rival['sprite_front']}" style="position: absolute; top: 75px; right: 20px;" width="100">
        </div>
        <div style="position: absolute; bottom: 30px; left: 50px; width: 240px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 10px; color: white; border-left: 5px solid #00d4ff;">
            <b>{current_ia['name']} (IA)</b>
            <div>{' / '.join(format_name(type_name) for type_name in current_ia['types'])}</div>
            <span style="float: right;">{int(st.session_state.env.hp_ia * 100)}%</span>
            <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; margin-top: 5px;"><div style="width: {st.session_state.env.hp_ia * 100}%; background: #4CAF50; height: 100%; border-radius: 5px;"></div></div>
            <img src="{current_ia['sprite_back']}" style="position: absolute; bottom: 60px; left: 20px;" width="120">
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
    st.caption(f"IA Types: {' / '.join(format_name(type_name) for type_name in current_ia['types'])}")
    st.caption(f"Rival Types: {' / '.join(format_name(type_name) for type_name in current_rival['types'])}")

with col_log:
    st.subheader("📜 Registro")
    with st.container(height=320):
        for entry in st.session_state.historial:
            st.write(entry)

with col_actions:
    if st.session_state.battle_finished:
        st.success(st.session_state.resultado)
    elif mode == "2. Desafío":
        st.subheader("🕹️ Tus Ataques")
        for idx, move in enumerate(current_rival["moves"]):
            label = f"💥 {move['name']} [{format_name(move['type'])}]"
            if st.button(label, key=f"at_{idx}", use_container_width=True, help=get_move_tooltip(move, current_ia)):
                ia_action = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
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
