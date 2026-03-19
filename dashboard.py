import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import os
import numpy as np
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Pokémon TFM - Final", layout="wide", page_icon="⚔️")

# --- COLORES POR TIPO ---
TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0", "Grass": "#78C850",
    "Electric": "#F8D030", "Ice": "#98D8D8", "Fighting": "#C03028", "Poison": "#A040A0",
    "Ground": "#E0C068", "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8", "Dark": "#705848",
    "Steel": "#B8B8D0", "Fairy": "#EE99AC"
}

# --- FUNCIONES DE API ---
@st.cache_data
def get_pokemon_data(name):
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}", timeout=3).json()
        stats = {s['stat']['name']: s['base_stat'] for s in r['stats']}
        moves_data = []
        for m_entry in r['moves']:
            if len(moves_data) >= 4: break
            m_info = requests.get(m_entry['move']['url']).json()
            if m_info['power']:
                moves_data.append({
                    "name": m_info['name'].replace("-", " ").title(),
                    "type": m_info['type']['name'].capitalize(),
                    "power": m_info['power']
                })
        return {
            "name": r['name'].capitalize(),
            "sprite_front": r['sprites']['versions']['generation-v']['black-white']['animated']['front_default'],
            "sprite_back": r['sprites']['versions']['generation-v']['black-white']['animated']['back_default'],
            "types": [t['type']['name'].capitalize() for t in r['types']],
            "stats": stats,
            "moves": moves_data if moves_data else [{"name": "Pound", "type": "Normal", "power": 40}]
        }
    except: return None

# --- ESTADO DE SESIÓN ---
if 'ia_name' not in st.session_state:
    st.session_state.ia_name = "Raichu"
    st.session_state.rival_name = "Dragonite"
    st.session_state.hp_ia, st.session_state.hp_rival = 100, 100
    st.session_state.historial = ["¡Módulo de combate listo!"]
    st.session_state.active_vfx = None
    st.session_state.rewards = [0]

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuración del Agente")
models_dir = 'models'
if not os.path.exists(models_dir): os.makedirs(models_dir)
zip_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
selected_checkpoint = st.sidebar.selectbox("📂 Checkpoint (Modelo RL):", zip_files if zip_files else ["Sin modelos"])

search_ia = st.sidebar.text_input("🔍 Buscar Pokémon IA:", value=st.session_state.ia_name)
if search_ia.capitalize() != st.session_state.ia_name:
    if get_pokemon_data(search_ia):
        st.session_state.ia_name = search_ia.capitalize()
        st.rerun()

# --- LÓGICA DE COMBATE ---
ia_data = get_pokemon_data(st.session_state.ia_name)
rival_data = get_pokemon_data(st.session_state.rival_name)
pokemon_color = TYPE_COLORS.get(ia_data['types'][0], "#31333F")

def combat_step(move_idx):
    move = ia_data['moves'][move_idx]
    st.session_state.active_vfx = move['type']
    dmg = int(move['power'] * 0.2)
    st.session_state.hp_rival = max(0, st.session_state.hp_rival - dmg)
    st.session_state.historial.insert(0, f"🤖 {st.session_state.ia_name} usó {move['name']}!")
    st.session_state.rewards.append(st.session_state.rewards[-1] + dmg)
    
    if st.session_state.hp_rival <= 0:
        st.balloons()
        st.session_state.hp_rival = 100
        st.session_state.rival_name = np.random.choice(["Charizard", "Mewtwo", "Blastoise"])

# --- RENDERIZADO DE LA ESCENA ---
vfx_banner = ""
if st.session_state.active_vfx:
    icon = "⚡" if st.session_state.active_vfx == "Electric" else "🔥" if st.session_state.active_vfx == "Fire" else "💥"
    vfx_banner = f'''
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 100;">
        <div style="background: rgba(0,0,0,0.7); color: white; padding: 12px 25px; border-radius: 12px; 
                    font-size: 32px; font-weight: bold; text-shadow: 2px 2px 10px black; border: 2px solid white;
                    text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-width: 220px;">
            {icon} {st.session_state.active_vfx.upper()} {icon}
        </div>
    </div>
    '''

battle_html = f'''
<div style="position: relative; background-image: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); 
            background-size: cover; background-position: center; height: 320px; width: 100%;
            border-radius: 15px; border: 4px solid #222; overflow: hidden; font-family: sans-serif;">
    
    <div style="position: absolute; top: 25px; right: 50px; text-align: center;">
        <div style="background: white; color: black; padding: 3px 12px; border-radius: 15px; 
                    font-weight: bold; border: 2px solid #333; font-size: 14px; margin-bottom: 5px;">
            {st.session_state.rival_name}: {st.session_state.hp_rival}%
        </div>
        <img src="{rival_data['sprite_front']}" width="120">
    </div>

    {vfx_banner}

    <div style="position: absolute; bottom: 20px; left: 50px; text-align: center;">
        <img src="{ia_data['sprite_back']}" width="160">
        <div style="background: white; color: black; padding: 3px 12px; border-radius: 15px; 
                    font-weight: bold; border: 2px solid #333; font-size: 14px; margin-top: 5px;">
            {st.session_state.ia_name}: {st.session_state.hp_ia}%
        </div>
    </div>
</div>
'''

components.html(battle_html, height=330)

# Barras de vida alineadas
c_hp1, c_hp2, c_hp3 = st.columns([1, 1, 1])
with c_hp1: st.progress(st.session_state.hp_ia / 100)
with c_hp3: st.progress(st.session_state.hp_rival / 100)

if st.session_state.active_vfx:
    time.sleep(0.8)
    st.session_state.active_vfx = None
    st.rerun()

# --- INTERFAZ DE CONTROL ---
st.write("###")
col_btns, col_switch, col_log = st.columns([1.5, 1, 1.2])

with col_btns:
    st.subheader("⚔️ Acciones")
    btns = st.columns(2)
    for i, m in enumerate(ia_data['moves']):
        if btns[i%2].button(f"{m['name']} ({m['type']})", key=f"btn_{i}", use_container_width=True):
            combat_step(i)
            st.rerun()

with col_switch:
    st.subheader("🔄 Selección")
    for p in ["Lucario", "Gengar", "Dragonite"]:
        if st.button(f"Cambiar a {p}", use_container_width=True):
            st.session_state.ia_name = p
            st.rerun()

with col_log:
    st.subheader("📜 Log")
    with st.container(height=160, border=True):
        for l in st.session_state.historial: st.write(f"• {l}")

# --- GRÁFICAS ---
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.write(f"**Recompensa Acumulada del Modelo**")
    st.area_chart(st.session_state.rewards, color=pokemon_color)
with g2:
    st.write("**Estadísticas Base del Pokémon**")
    st.bar_chart(pd.Series(ia_data['stats']), color=pokemon_color)