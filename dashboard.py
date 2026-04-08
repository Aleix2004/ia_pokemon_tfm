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
        ''', (str(modelo), ganador, turnos, ia_vivos, riv_vivos, 0.0))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error SQL: {e}")

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA Pokémon TFM - Dashboard Pro", layout="wide", page_icon="🐲")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- 2. MOTOR POKEAPI ---
@st.cache_data
def get_item_data(item_name):
    if not item_name: return None
    try:
        url = f"https://pokeapi.co/api/v2/item/{item_name.lower().replace(' ', '-')}"
        r = requests.get(url, timeout=5).json()
        return {"name": r['name'].replace("-", " ").capitalize(), "sprite": r['sprites']['default']}
    except: return None

@st.cache_data
def get_pokemon_data(name_or_id, item_name="Life Orb", is_shiny=None):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        r = requests.get(url, timeout=5).json()
        if is_shiny is None: is_shiny = random.random() < 0.10
        api_stats = {s['stat']['name']: s['base_stat'] for s in r['stats']}
        animated = r['sprites']['versions']['generation-v']['black-white']['animated']
        
        s_key = 'front_shiny' if is_shiny else 'front_default'
        sb_key = 'back_shiny' if is_shiny else 'back_default'
        img_f = animated[s_key] if animated[s_key] else r['sprites'][s_key]
        img_b = animated[sb_key] if animated[sb_key] else r['sprites'][sb_key]
        
        return {
            "name": r['name'].capitalize(),
            "sprite_front": img_f, "sprite_back": img_b,
            "stats": {
                "Hp": api_stats.get("hp", 0), 
                "Attack": api_stats.get("attack", 0), 
                "Defense": api_stats.get("defense", 0),
                "Sp. Attack": api_stats.get("special-attack", 0),
                "Sp. Defense": api_stats.get("special-defense", 0),
                "Speed": api_stats.get("speed", 0)
            },
            "item": get_item_data(item_name),
            "debilitado": False,
            "moves": [{"name": m['move']['name'].replace("-", " ").capitalize()} for m in r['moves'][:4]]
        }
    except: return None

# --- 3. INICIALIZACIÓN ---
if 'game_started' not in st.session_state:
    st.session_state.update({
        'game_started': False, 'battle_finished': False, 'resultado': "",
        'active_ia': 0, 'active_rival': 0, 'historial': [],
        'env': PokemonEnv(), 'db_logged': False, 'loaded_model': None, 'current_model_path': ""
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
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "¡GANASTE! 🎉"

    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]['debilitado'] = True
        siguiente = next((i for i, p in enumerate(st.session_state.team_ia) if not p['debilitado']), None)
        if siguiente is not None:
            st.session_state.active_ia = siguiente; st.session_state.env.hp_ia = 1.0
        else: 
            st.session_state.battle_finished = True
            st.session_state.resultado = "PERDISTE 💀"
    
    if st.session_state.battle_finished and not st.session_state.db_logged:
        guardar_batalla_sql(st.session_state.resultado, sel_model_name, st.session_state.historial)
        st.session_state.db_logged = True

# --- 4. SELECCIÓN ---
if not st.session_state.game_started and not st.session_state.battle_finished:
    st.title("🧪 Configuración TFM - Big Data & IA")
    def build_card(defaults, prefix, color):
        team = []
        with st.container(border=True):
            st.markdown(f"<h3 style='color:{color}; text-align:center;'>EQUIPO {prefix}</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, name in enumerate(defaults):
                with cols[i%3]:
                    pk = st.text_input(f"Pkmn {i+1}", value=name, key=f"p{prefix}{i}")
                    it = st.text_input(f"Item {i+1}", value="Life Orb", key=f"i{prefix}{i}")
                    p_d = get_pokemon_data(pk, it)
                    if p_d:
                        with st.container(border=True):
                            c1, c2 = st.columns([2,1])
                            c1.image(p_d['sprite_front'], width=60)
                            if p_d['item']: c2.image(p_d['item']['sprite'], width=30)
                            st.caption(f"<center><b>{p_d['name']}</b></center>", unsafe_allow_html=True)
                        team.append(p_d)
        return team
    c1, c2 = st.columns(2)
    with c1: t_ia = build_card(["Charizard", "Gengar", "Lucario", "Metagross", "Rayquaza", "Sceptile"], "IA", "#00d4ff")
    with c2: t_riv = build_card(["Mewtwo", "Arceus", "Garchomp", "Kyogre", "Tyranitar", "Zoroark"], "RIVAL", "#ff4b4b")
    if st.button("🔥 INICIAR", type="primary", use_container_width=True):
        st.session_state.update({'team_ia': t_ia, 'team_rival': t_riv, 'game_started': True})
        st.rerun()
    st.stop()

# --- 5. SIDEBAR ---
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
with st.sidebar:
    st.header("⚙️ Configuración")
    model_list = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".zip"):
                model_list.append(os.path.relpath(os.path.join(root, file), MODELS_DIR))
    
    sel_model = st.selectbox("Elegir Modelo ZIP:", sorted(model_list))
    auto = st.toggle("Modo Automático", value=False)
    vel = st.slider("Velocidad", 0.1, 2.0, 0.5)
    
    st.divider()
    # MOSTRAR STATS ACTUALES (Incluyendo Especiales)
    st.subheader(f"📊 Stats Actual: {curr_ia['name']}")
    df_stats = pd.DataFrame.from_dict(curr_ia['stats'], orient='index', columns=['Valor'])
    st.table(df_stats)

# --- 6. ARENA ---
curr_riv = st.session_state.team_rival[st.session_state.active_rival]
hp_ia, hp_rival = int(st.session_state.env.hp_ia*100), int(st.session_state.env.hp_rival*100)

st.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 260px; border-radius: 15px; position: relative; border: 2px solid #333; margin-bottom: 20px;">
    <div style="position: absolute; top: 20px; right: 40px; width: 170px; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 10px; border: 1px solid #ff4b4b; color: white;">
        <b>{curr_riv['name']}</b> <span style="float: right; color: #4CAF50;">{hp_rival}%</span>
        <div style="width: 100%; background: #333; height: 8px; border-radius: 4px; margin-top: 5px;"><div style="width: {hp_rival}%; background: #4CAF50; height: 100%;"></div></div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 50px; right: 5px;" width="80">
    </div>
    <div style="position: absolute; bottom: 20px; left: 40px; width: 170px; background: rgba(0,0,0,0.8); padding: 8px; border-radius: 10px; border: 1px solid #00d4ff; color: white;">
        <b>{curr_ia['name']}</b> <span style="float: right; color: #4CAF50;">{hp_ia}%</span>
        <div style="width: 100%; background: #333; height: 8px; border-radius: 4px; margin-top: 5px;"><div style="width: {hp_ia}%; background: #4CAF50; height: 100%;"></div></div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 45px; left: 5px;" width="110">
    </div>
</div>
''')

# --- MONITOR DE INTELIGENCIA GLOBAL ---
with st.expander("👁️ Monitor de Inteligencia: Estado Global de la Batalla", expanded=True):
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.markdown("<p style='text-align:center; color:#00d4ff;'><b>🔵 TU EQUIPO (IA)</b></p>", unsafe_allow_html=True)
        cols_ia = st.columns(6)
        for i, p in enumerate(st.session_state.team_ia):
            with cols_ia[i]:
                op = "0.3" if p.get('debilitado') else "1"
                s = p['stats']
                it_img = f'<img src="{p["item"]["sprite"]}" width="15">' if p['item'] else ""
                st.markdown(f"""
                    <div style="text-align: center; opacity: {op}; font-size: 0.65rem; border: 1px solid #333; border-radius: 5px; padding: 2px;">
                        <img src="https://img.pokemondb.net/sprites/black-white/anim/back-normal/{p['name'].lower()}.gif" width="40"><br>
                        <strong>{p['name']}</strong> {it_img}<br>
                        <span style="color: #ff4b4b;">❤️{s['Hp']}</span> <span style="color: #1c83e1;">⚔️{s['Attack']}</span><br>
                        <span style="color: #00c0f2;">🛡️{s['Defense']}</span> <span style="color: #fca311;">⚡{s['Speed']}</span>
                    </div>
                """, unsafe_allow_html=True)

    with col_der:
        st.markdown("<p style='text-align:center; color:#ff4b4b;'><b>🔴 EQUIPO RIVAL</b></p>", unsafe_allow_html=True)
        cols_riv = st.columns(6)
        for i, p in enumerate(st.session_state.team_rival):
            with cols_riv[i]:
                op = "0.3" if p.get('debilitado') else "1"
                s = p['stats']
                it_img = f'<img src="{p["item"]["sprite"]}" width="15">' if p['item'] else ""
                st.markdown(f"""
                    <div style="text-align: center; opacity: {op}; font-size: 0.65rem; border: 1px solid #333; border-radius: 5px; padding: 2px;">
                        <img src="https://img.pokemondb.net/sprites/black-white/anim/normal/{p['name'].lower()}.gif" width="40"><br>
                        <strong>{p['name']}</strong> {it_img}<br>
                        <span style="color: #ff4b4b;">❤️{s['Hp']}</span> <span style="color: #1c83e1;">⚔️{s['Attack']}</span><br>
                        <span style="color: #00c0f2;">🛡️{s['Defense']}</span> <span style="color: #fca311;">⚡{s['Speed']}</span>
                    </div>
                """, unsafe_allow_html=True)

# --- 7. CONTROLES ---
c_at, c_log, c_sw = st.columns([1, 2, 1])
with c_at:
    st.subheader("⚔️ Ataques")
    if not auto:
        for i, m in enumerate(curr_ia['moves']):
            if st.button(m['name'], key=f"at{i}", use_container_width=True):
                combat_step(i, sel_model); st.rerun()

with c_log:
    st.subheader("📜 Registro")
    with st.container(height=250, border=True):
        for e in st.session_state.historial: st.write(e)

with c_sw:
    st.subheader("🔄 Cambios")
    for idx, p in enumerate(st.session_state.team_ia):
        if st.button(p['name'], key=f"sw{idx}", disabled=p['debilitado'] or idx==st.session_state.active_ia, use_container_width=True):
            st.session_state.active_ia = idx; st.session_state.env.hp_ia = 1.0; st.rerun()

# --- 8. MODO AUTO ---
if auto and not st.session_state.battle_finished:
    if sel_model:
        model_path = os.path.join(MODELS_DIR, sel_model)
        if st.session_state.current_model_path != model_path:
            st.session_state.loaded_model = PPO.load(model_path)
            st.session_state.current_model_path = model_path
        
        action, _ = st.session_state.loaded_model.predict(st.session_state.env._get_obs())
        combat_step(int(action), sel_model)
        time.sleep(vel); st.rerun() 

# --- 9. RESULTADOS ---
if st.session_state.battle_finished:
    st.success(st.session_state.resultado)
    try:
        conn = sqlite3.connect('pokemon_bigdata.db')
        df = pd.read_sql_query("SELECT * FROM battle_logs ORDER BY timestamp DESC LIMIT 5", conn)
        st.subheader("📊 Histórico SQL")
        st.dataframe(df, use_container_width=True)
        conn.close()
    except: pass
    if st.button("🔄 Nueva Simulación", use_container_width=True):
        st.session_state.clear(); st.rerun()