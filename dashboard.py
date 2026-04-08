import streamlit as st
import requests
import pandas as pd
import os
import time
import numpy as np
import random
import sqlite3
from stable_baselines3 import PPO
from src.env.pokemon_env import PokemonEnv 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Pokémon AI TFM Dashboard", page_icon="🧪")

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

# --- MOTOR POKEAPI MEJORADO ---
@st.cache_data
def get_item_data(item_name):
    if not item_name: return None
    try:
        url = f"https://pokeapi.co/api/v2/item/{item_name.lower().replace(' ', '-')}"
        r = requests.get(url, timeout=5).json()
        return {"name": r['name'].capitalize(), "sprite": r['sprites']['default']}
    except: return None

@st.cache_data
def get_pokemon_data(name_or_id, item_name="Life Orb"):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        r = requests.get(url, timeout=5).json()
        api_stats = {s['stat']['name']: s['base_stat'] for s in r['stats']}
        animated = r['sprites']['versions']['generation-v']['black-white']['animated']
        img_f = animated['front_default'] if animated['front_default'] else r['sprites']['front_default']
        img_b = animated['back_default'] if animated['back_default'] else r['sprites']['back_default']
        
        return {
            "name": r['name'].capitalize(),
            "sprite_front": img_f,
            "sprite_back": img_b,
            "stats": {
                "hp": api_stats.get("hp", 0),
                "atk": api_stats.get("attack", 0),
                "def": api_stats.get("defense", 0),
                "sp_atk": api_stats.get("special-attack", 0),
                "sp_def": api_stats.get("special-defense", 0),
                "spd": api_stats.get("speed", 0)
            },
            "item": get_item_data(item_name),
            "debilitado": False,
            "moves": [{"name": m['move']['name'].replace("-", " ").capitalize()} for m in r['moves'][:4]]
        }
    except: return None

# --- INICIALIZACIÓN ---
if 'game_started' not in st.session_state:
    st.session_state.update({
        'game_started': False, 
        'battle_finished': False, 
        'resultado': "",
        'active_ia': 0, 
        'active_rival': 0, 
        'historial': [],
        'env': PokemonEnv(), 
        'loaded_model': None, 
        'current_model_path': "",
        'auto_enabled': False
    })

def predict_action_compatible(model, env):
    obs, _ = env.reset() if not hasattr(env, 'hp_ia') else (env._get_obs(), None)
    try:
        return model.predict(obs)[0]
    except:
        return random.randint(0, 3)

def combat_step(action_ia, action_rival=None, sel_model_name="N/A"):
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    curr_riv = st.session_state.team_rival[st.session_state.active_rival]
    old_hp_riv, old_hp_ia = st.session_state.env.hp_rival, st.session_state.env.hp_ia

    ia_move_name = curr_ia['moves'][int(action_ia)]['name']
    obs, reward, terminated, truncated, info = st.session_state.env.step(action_ia, action_rival=action_rival, ia_move_name=ia_move_name)
    
    dmg_to_riv = max(0, (old_hp_riv - st.session_state.env.hp_rival) * 100)
    dmg_to_ia = max(0, (old_hp_ia - st.session_state.env.hp_ia) * 100)

    st.session_state.historial.insert(0, f"🔴 **{curr_riv['name']}**: -{dmg_to_ia:.1f}% ({info['rival_move']})")
    st.session_state.historial.insert(0, f"⚔️ **{curr_ia['name']}** (IA): -{dmg_to_riv:.1f}% ({info['ia_move']})")

    if st.session_state.env.hp_rival <= 0:
        st.session_state.team_rival[st.session_state.active_rival]['debilitado'] = True
        if st.session_state.active_rival < 5:
            st.session_state.active_rival += 1
            st.session_state.env.hp_rival = 1.0
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "🏆 ¡VICTORIA DE LA IA!"

    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]['debilitado'] = True
        siguiente = next((i for i, p in enumerate(st.session_state.team_ia) if not p['debilitado']), None)
        if siguiente is not None:
            st.session_state.active_ia = siguiente
            st.session_state.env.hp_ia = 1.0
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "💀 LA IA HA SIDO DERROTADA"

# --- PANTALLA DE SELECCIÓN ---
if not st.session_state.game_started:
    st.title("🧪 Configuración de Equipos - TFM AI")
    
    def render_team_selection(title, defaults, key_prefix):
        st.subheader(title)
        team = []
        cols = st.columns(3)
        for i in range(6):
            with cols[i % 3]:
                name = st.text_input(f"Pokémon {i+1}", value=defaults[i], key=f"{key_prefix}_n_{i}")
                item = st.text_input(f"Objeto {i+1}", value="Life Orb", key=f"{key_prefix}_i_{i}")
                data = get_pokemon_data(name, item)
                if data:
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 1])
                        c1.image(data['sprite_front'], width=70)
                        if data['item']: c2.image(data['item']['sprite'], width=40, caption=data['item']['name'])
                        st.caption(f"**{data['name']}**")
                    team.append(data)
        return team

    col_ia, col_riv = st.columns(2)
    with col_ia:
        t_ia = render_team_selection("🤖 Equipo IA", ["Mewtwo", "Rayquaza", "Kyogre", "Groudon", "Metagross", "Sceptile"], "ia")
    with col_riv:
        t_riv = render_team_selection("👤 Equipo Rival", ["Charizard", "Blastoise", "Venusaur", "Gengar", "Lucario", "Tyranitar"], "riv")
    
    if st.button("🔥 INICIAR COMBATE", type="primary", use_container_width=True):
        if len(t_ia) == 6 and len(t_riv) == 6:
            # --- LIMPIEZA PREVENTIVA DE SQL ---
            try:
                conn = sqlite3.connect('pokemon_bigdata.db')
                conn.execute("DROP TABLE IF EXISTS v_logs")
                conn.execute('''CREATE TABLE v_logs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                ia_move_name TEXT, rival_move TEXT,
                                hp_ia REAL, hp_rival REAL, reward REAL)''')
                conn.close()
            except: pass

            st.session_state.update({'team_ia': t_ia, 'team_rival': t_riv, 'game_started': True})
            st.rerun()
        else:
            st.error("Asegúrate de que todos los Pokémon hayan cargado correctamente.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕹️ Panel de Control")
    modo = st.radio("Modo:", ["1. Simulación", "2. Desafío"])
    
    # NUEVA FUNCIÓN: Búsqueda recursiva de archivos .zip
    model_list = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".zip"):
                # Obtenemos la ruta relativa para que el selector sea limpio
                relative_path = os.path.relpath(os.path.join(root, f), MODELS_DIR)
                model_list.append(relative_path)
    
    if model_list:
        # Ordenamos la lista para que sea más fácil encontrar los modelos
        sel_model = st.selectbox("Modelo PPO:", sorted(model_list))
        
        # Ruta completa para cargar el modelo
        model_path = os.path.join(MODELS_DIR, sel_model)
        
        # Carga inteligente: Solo recarga si el modelo seleccionado cambia
        if st.session_state.current_model_path != model_path:
            try:
                st.session_state.loaded_model = PPO.load(model_path)
                st.session_state.current_model_path = model_path
                st.success(f"Cerebro cargado: {sel_model}")
            except Exception as e:
                st.error(f"Error al cargar {sel_model}: {e}")
    else:
        st.error(f"No se encontraron modelos .zip en la ruta: {MODELS_DIR}")

    auto = st.toggle("Auto-Play", value=st.session_state.auto_enabled)
    st.session_state.auto_enabled = auto 
    
    vel = st.slider("Velocidad", 0.1, 2.0, 0.5)
    
    st.divider()
    # Mostrar Stats del Pokémon actual de la IA
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    st.subheader(f"📊 Stats: {curr_ia['name']}")
    st.table(pd.Series(curr_ia['stats']))

# --- ARENA ---
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
curr_riv = st.session_state.team_rival[st.session_state.active_rival]

st.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 300px; border-radius: 20px; position: relative; border: 3px solid #444;">
    <div style="position: absolute; top: 30px; right: 50px; width: 220px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 10px; color: white; border-left: 5px solid #ff4b4b;">
        <b>{curr_riv['name']}</b> <span style="float: right;">{int(st.session_state.env.hp_rival*100)}%</span>
        <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; margin-top: 5px;"><div style="width: {st.session_state.env.hp_rival*100}%; background: #4CAF50; height: 100%; border-radius: 5px;"></div></div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 60px; right: 20px;" width="100">
    </div>
    <div style="position: absolute; bottom: 30px; left: 50px; width: 220px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 10px; color: white; border-left: 5px solid #00d4ff;">
        <b>{curr_ia['name']} (IA)</b> <span style="float: right;">{int(st.session_state.env.hp_ia*100)}%</span>
        <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; margin-top: 5px;"><div style="width: {st.session_state.env.hp_ia*100}%; background: #4CAF50; height: 100%; border-radius: 5px;"></div></div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 60px; left: 20px;" width="120">
    </div>
</div>
''')

# Monitor Global (Sprites pequeños)
m_ia, m_riv = st.columns(2)
with m_ia:
    cols = st.columns(6)
    for i, p in enumerate(st.session_state.team_ia):
        cols[i].markdown(f'<div style="text-align:center; opacity:{"1" if not p["debilitado"] else "0.3"}; border: 2px solid {"#00d4ff" if i==st.session_state.active_ia else "transparent"}; border-radius:10px;"><img src="{p["sprite_front"]}" width="45"></div>', unsafe_allow_html=True)
with m_riv:
    cols = st.columns(6)
    for i, p in enumerate(st.session_state.team_rival):
        cols[i].markdown(f'<div style="text-align:center; opacity:{"1" if not p["debilitado"] else "0.3"}; border: 2px solid {"#ff4b4b" if i==st.session_state.active_rival else "transparent"}; border-radius:10px;"><img src="{p["sprite_front"]}" width="45"></div>', unsafe_allow_html=True)

st.divider()

# Lógica de Combate y Columnas Inferiores
c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    st.subheader("📊 Comparativa")
    st.table(pd.DataFrame({
        "IA (Aliado)": curr_ia['stats'],
        "Rival": curr_riv['stats']
    }))

with c2:
    st.subheader("📜 Registro")
    with st.container(height=300):
        for h in st.session_state.historial: st.write(h)

with c3:
    if st.session_state.battle_finished:
        st.success(st.session_state.resultado)
    elif modo == "2. Desafío":
        st.subheader("🕹️ Tus Ataques")
        for i, m in enumerate(curr_riv['moves']):
            if st.button(f"💥 {m['name']}", key=f"at_{i}", use_container_width=True):
                ia_act = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
                combat_step(ia_act, action_rival=i)
                st.rerun()
    else:
        if not auto:
            st.warning("⏸️ Simulación pausada.")
        else:
            if not st.session_state.battle_finished:
                ia_act = predict_action_compatible(st.session_state.loaded_model, st.session_state.env)
                combat_step(ia_act)
                time.sleep(vel)
                st.rerun()

# --- REPORTE SQL FINAL ---
if st.session_state.battle_finished:
    st.divider()
    st.header("📊 Informe Analítico Post-Combate")
    
    try:
        conn = sqlite3.connect('pokemon_bigdata.db')
        df_hp = pd.read_sql("SELECT id, hp_ia, hp_rival FROM v_logs ORDER BY id ASC", conn)
        
        if not df_hp.empty:
            st.subheader("📈 Evolución de Vitalidad")
            st.line_chart(df_hp.set_index('id'))
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.subheader("⚔️ Movimientos IA")
                df_ia = pd.read_sql("SELECT ia_move_name as Movimiento, COUNT(*) as Usos FROM v_logs GROUP BY ia_move_name", conn)
                st.dataframe(df_ia, use_container_width=True)
            with c_s2:
                st.subheader("🛡️ Movimientos Rival")
                df_riv = pd.read_sql("SELECT rival_move as Movimiento, COUNT(*) as Usos FROM v_logs GROUP BY rival_move", conn)
                st.dataframe(df_riv, use_container_width=True)
        conn.close()
    except Exception as e:
        st.error(f"Error cargando informe: {e}")

    if st.button("🔄 REINICIAR TODO", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()