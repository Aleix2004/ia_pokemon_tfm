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

# --- 0. FUNCIONES DE BIG DATA (SQL) ---
def guardar_batalla_sql(resultado, modelo, historial):
    try:
        conn = sqlite3.connect('pokemon_bigdata.db')
        cursor = conn.cursor()
        
        ganador = "IA" if "¡GANASTE!" in resultado else "RIVAL"
        turnos = len([e for e in historial if "⚔️" in e])
        ia_vivos = sum(1 for p in st.session_state.team_ia if not p['debilitado'])
        riv_vivos = sum(1 for p in st.session_state.team_rival if not p['debilitado'])
        
        cursor.execute('''
            INSERT INTO battle_logs 
            (model_name, winner, turns, ia_pokemon_left, rival_pokemon_left, total_damage_dealt)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (modelo, ganador, turnos, ia_vivos, riv_vivos, 0.0))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error SQL: {e}")

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA Pokémon TFM - Big Data Edition", layout="wide", page_icon="🐲")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- 2. MOTOR POKEAPI ---
@st.cache_data
def get_pokemon_data(name_or_id, is_shiny=None):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        r = requests.get(url, timeout=5).json()
        if is_shiny is None: is_shiny = random.random() < 0.10
        animated = r['sprites']['versions']['generation-v']['black-white']['animated']
        s_key = 'front_shiny' if is_shiny else 'front_default'
        sb_key = 'back_shiny' if is_shiny else 'back_default'
        img_f = animated[s_key] if animated[s_key] else r['sprites'][s_key]
        img_b = animated[sb_key] if animated[sb_key] else r['sprites'][sb_key]
        stats = {s['stat']['name'].replace("-", " ").title(): s['base_stat'] for s in r['stats']}
        species_data = requests.get(r['species']['url']).json()
        megas = [v['pokemon']['name'] for v in species_data['varieties'] if "mega" in v['pokemon']['name']]
        return {
            "name": r['name'].capitalize(), "species_name": species_data['name'],
            "sprite_front": img_f, "sprite_back": img_b, "is_shiny": is_shiny, 
            "available_megas": megas, "stats": stats, "debilitado": False,
            "moves": [{"name": m['move']['name'].replace("-", " ").capitalize()} for m in r['moves'][:4]]
        }
    except: return None

@st.cache_data
def get_item_data(item_name):
    if not item_name: return None
    try:
        url = f"https://pokeapi.co/api/v2/item/{item_name.lower().replace(' ', '-')}"
        r = requests.get(url, timeout=5).json()
        return {
            "name": r['name'].replace("-", " ").capitalize(), "sprite": r['sprites']['default'],
            "effect": r['effect_entries'][0]['short_effect'] if r['effect_entries'] else "Sin efecto"
        }
    except: return None

# --- 3. LÓGICA DE COMBATE ---
if 'game_started' not in st.session_state:
    st.session_state.update({
        'game_started': False, 'battle_finished': False, 'resultado': "",
        'active_ia': 0, 'active_rival': 0, 'damage_data': {}, 'historial': [],
        'env': PokemonEnv(), 'db_logged': False
    })

def combat_step(action_idx, sel_model_name):
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    curr_riv = st.session_state.team_rival[st.session_state.active_rival]
    old_hp_riv = st.session_state.env.hp_rival
    old_hp_ia = st.session_state.env.hp_ia

    move_name = curr_ia['moves'][action_idx]['name'] if action_idx < 4 else "Cambio"
    obs, reward, terminated, truncated, info = st.session_state.env.step(action_idx, ia_move_name=move_name)
    
    dmg_to_riv = max(0, (old_hp_riv - st.session_state.env.hp_rival) * 100)
    dmg_to_ia = max(0, (old_hp_ia - st.session_state.env.hp_ia) * 100)

    st.session_state.historial.insert(0, f"⚔️ **{curr_ia['name']}** usó **{info['ia_move']}**: -{dmg_to_riv:.1f}%")
    
    if dmg_to_ia > 0 or "✨" in info['rival_move']:
        st.session_state.historial.insert(0, f"🔴 **{curr_riv['name']}** usó **{info['rival_move']}**: -{dmg_to_ia:.1f}%")

    if st.session_state.env.hp_rival <= 0:
        st.session_state.team_rival[st.session_state.active_rival]['debilitado'] = True
        if st.session_state.active_rival < 5:
            st.session_state.active_rival += 1
            st.session_state.env.hp_rival = 1.0
            st.session_state.historial.insert(0, f"💥 **RIVAL KO**: Entra {st.session_state.team_rival[st.session_state.active_rival]['name']}")
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "¡GANASTE! 🎉"

    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]['debilitado'] = True
        siguiente = next((i for i, p in enumerate(st.session_state.team_ia) if not p['debilitado']), None)
        if siguiente is not None:
            st.session_state.active_ia = siguiente; st.session_state.env.hp_ia = 1.0
            st.session_state.historial.insert(0, f"💀 **TUYA KO**: Sale {st.session_state.team_ia[siguiente]['name']}")
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "PERDISTE 💀"
    
    # Registro en SQL al terminar
    if st.session_state.battle_finished and not st.session_state.db_logged:
        guardar_batalla_sql(st.session_state.resultado, sel_model_name, st.session_state.historial)
        st.session_state.db_logged = True

# --- 4. PANTALLA SELECCIÓN ---
if not st.session_state.game_started and not st.session_state.battle_finished:
    st.title("🧪 Configuración TFM - Big Data & IA")
    def build_card(defaults, prefix, color):
        team = []
        with st.container(border=True):
            st.markdown(f"<h3 style='text-align: center; color: {color};'>EQUIPO {prefix}</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, name in enumerate(defaults):
                with cols[i%3]:
                    pk = st.text_input(f"Pkmn {i+1}", value=name, key=f"p{prefix}{i}")
                    it = st.text_input(f"Item {i+1}", value="Life Orb", key=f"i{prefix}{i}")
                    p_d = get_pokemon_data(pk); i_d = get_item_data(it)
                    if p_d:
                        p_d['item'] = i_d
                        with st.container(border=True):
                            c1, c2 = st.columns([2,1])
                            c1.image(p_d['sprite_front'], width=65)
                            if i_d: c2.image(i_d['sprite'], width=30)
                            st.caption(f"<center><b>{p_d['name']}</b></center>", unsafe_allow_html=True)
                        team.append(p_d)
        return team
    c1, c2 = st.columns(2)
    with c1: t_ia = build_card(["Charizard", "Gengar", "Lucario", "Metagross", "Rayquaza", "Sceptile"], "IA", "#00d4ff")
    with c2: t_riv = build_card(["Mewtwo", "Arceus", "Garchomp", "Kyogre", "Tyranitar", "Zoroark"], "RIVAL", "#ff4b4b")
    if st.button("🔥 INICIAR", use_container_width=True, type="primary"):
        st.session_state.update({'team_ia': t_ia, 'team_rival': t_riv, 'game_started': True})
        st.session_state.env.reset(); st.rerun()
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    model_list = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".zip"):
                model_list.append(os.path.relpath(os.path.join(root, file), MODELS_DIR))
    
    sel_model = st.selectbox("Modelo:", sorted(model_list))
    auto = st.toggle("Modo Automático", value=False)
    vel = st.slider("Velocidad", 0.1, 2.0, 0.5)
    
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    st.divider(); st.subheader("📊 Stats Actual"); st.table(pd.Series(curr_ia['stats']))

# --- 6. ARENA ---
ia_vivos = sum(1 for p in st.session_state.team_ia if not p['debilitado'])
riv_vivos = sum(1 for p in st.session_state.team_rival if not p['debilitado'])
riv_balls = " ".join([f"<span style='color: #ff4b4b;'>●</span>" if i < riv_vivos else "<span style='color: #555;'>○</span>" for i in range(6)])
ia_balls = " ".join([f"<span style='color: #00d4ff;'>●</span>" if i < ia_vivos else "<span style='color: #555;'>○</span>" for i in range(6)])
curr_riv = st.session_state.team_rival[st.session_state.active_rival]
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
hp_ia, hp_rival = int(st.session_state.env.hp_ia*100), int(st.session_state.env.hp_rival*100)

st.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 280px; border-radius: 15px; position: relative; border: 2px solid #333; margin-bottom: 20px;">
    <div style="position: absolute; top: 20px; right: 40px; width: 180px; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 10px; border: 1px solid #ff4b4b;">
        <div style="display: flex; justify-content: space-between; color: white;"><b>{curr_riv['name']}</b> <span style="color: #4CAF50;">{hp_rival}%</span></div>
        <div style="font-size: 16px;">{riv_balls}</div>
        <div style="width: 100%; background: #333; height: 8px; border-radius: 4px; overflow: hidden;"><div style="width: {hp_rival}%; background: #4CAF50; height: 100%;"></div></div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 55px; right: 10px;" width="80">
    </div>
    <div style="position: absolute; bottom: 20px; left: 40px; width: 180px; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 10px; border: 1px solid #00d4ff;">
        <div style="display: flex; justify-content: space-between; color: white;"><b>{curr_ia['name']}</b> <span style="color: #4CAF50;">{hp_ia}%</span></div>
        <div style="font-size: 16px;">{ia_balls}</div>
        <div style="width: 100%; background: #333; height: 8px; border-radius: 4px; overflow: hidden;"><div style="width: {hp_ia}%; background: #4CAF50; height: 100%;"></div></div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 50px; left: 10px;" width="110">
    </div>
</div>
''')

with st.expander("👁️ Monitor de Inteligencia: Estado del Equipo Rival"):
    cols_riv = st.columns(6)
    for i, p_riv in enumerate(st.session_state.team_rival):
        with cols_riv[i]:
            if p_riv.get('debilitado', False): st.write("💀")
            else:
                st.image(p_riv['sprite_front'], width=50)
                st.caption(f"{p_riv['name']}")

# --- 7. CONTROLES ---
col_acc, col_log, col_sw = st.columns([1, 2, 1])
with col_acc:
    st.subheader("⚔️ Ataques")
    if not auto:
        for i, m in enumerate(curr_ia['moves']):
            if st.button(m['name'], key=f"at{i}", use_container_width=True):
                combat_step(i, sel_model); st.rerun()

with col_log:
    st.subheader("📜 Registro")
    with st.container(height=280, border=True):
        for e in st.session_state.historial:
            st.write(e)

with col_sw:
    st.subheader("🔄 Cambios")
    for idx, p in enumerate(st.session_state.team_ia):
        if st.button(p['name'], key=f"sw{idx}", use_container_width=True, disabled=p.get('debilitado') or idx==st.session_state.active_ia):
            st.session_state.active_ia = idx; st.session_state.env.hp_ia = 1.0; st.rerun()

# --- 8. MODO AUTO ---
if auto and not st.session_state.battle_finished:
    if sel_model:
        model = PPO.load(os.path.join(MODELS_DIR, sel_model))
        action, _ = model.predict(st.session_state.env._get_obs())
        combat_step(int(action), sel_model)
        time.sleep(vel); st.rerun() 

# --- 9. RESULTADOS Y BIG DATA ---
if st.session_state.battle_finished:
    st.success(st.session_state.resultado)
    
    # Visualización de la Base de Datos SQL (Punto 4 del PDF)
    st.divider()
    st.subheader("📊 Histórico de Batallas (Capa SQL Big Data)")
    try:
        conn = sqlite3.connect('pokemon_bigdata.db')
        df_logs = pd.read_sql_query("SELECT * FROM battle_logs ORDER BY timestamp DESC LIMIT 5", conn)
        st.dataframe(df_logs, use_container_width=True)
        conn.close()
    except:
        st.info("Juega una batalla para ver el registro SQL.")

    if st.button("🔄 Reiniciar Simulación", use_container_width=True):
        st.session_state.clear(); st.rerun()