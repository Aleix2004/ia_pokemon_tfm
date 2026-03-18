import streamlit as st
import requests
import pandas as pd
import os
import numpy as np
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Pokémon TFM - Final Edition", layout="wide", page_icon="⚔️")

# --- DICCIONARIO DE COLORES POR TIPO ---
TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0", "Grass": "#78C850",
    "Electric": "#F8D030", "Ice": "#98D8D8", "Fighting": "#C03028", "Poison": "#A040A0",
    "Ground": "#E0C068", "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8", "Dark": "#705848",
    "Steel": "#B8B8D0", "Fairy": "#EE99AC"
}

# --- FUNCIONES DE API (PokeAPI) ---
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

# --- ESTILOS CSS Y VFX ---
st.markdown(f"""
<style>
    .stButton>button {{ border-radius: 8px; font-weight: bold; text-shadow: 1px 1px 2px #000; color: white !important; height: 3.5em; border: 2px solid #222; }}
    
    /* VFX: Lanzallamas */
    .vfx-fire {{
        position: absolute; bottom: 110px; left: 160px; height: 50px;
        background: radial-gradient(circle, #ff4500, #ff8c00, transparent);
        filter: blur(10px); border-radius: 50%; z-index: 999;
        animation: flame_anim 0.7s forwards;
    }}
    @keyframes flame_anim {{ 0% {{ width: 0; opacity: 1; }} 100% {{ width: 380px; opacity: 0; transform: translateX(100px); }} }}

    /* VFX: Rayo */
    .vfx-electric {{
        position: absolute; top: 40px; right: 110px; width: 10px; height: 240px;
        background: #ffff00; box-shadow: 0 0 25px #ffff00; transform: rotate(-45deg);
        z-index: 999; animation: bolt_anim 0.15s 4;
    }}
    @keyframes bolt_anim {{ 0%, 100% {{ opacity: 0; }} 50% {{ opacity: 1; }} }}

    /* VFX: Golpe Físico */
    .vfx-hit {{
        position: absolute; top: 80px; right: 120px; font-size: 60px; z-index: 999;
        animation: hit_anim 0.4s ease-out;
    }}
    @keyframes hit_anim {{ 0% {{ transform: scale(0.5); opacity: 0; }} 50% {{ transform: scale(1.5); opacity: 1; }} 100% {{ transform: scale(1); opacity: 0; }} }}
</style>
""", unsafe_allow_html=True)

# --- ESTADO DE SESIÓN ---
if 'ia_name' not in st.session_state:
    st.session_state.ia_name = "Arceus"
    st.session_state.rival_name = "Blastoise"
    st.session_state.hp_ia, st.session_state.hp_rival = 100, 100
    st.session_state.historial = ["¡Duelo listo!"]
    st.session_state.active_vfx = None
    st.session_state.rewards = [0]

# --- SIDEBAR ---
st.sidebar.header("🕹️ Panel de Control")
search_ia = st.sidebar.text_input("🔍 Buscar Pokémon IA:", value=st.session_state.ia_name)
if search_ia.capitalize() != st.session_state.ia_name:
    if get_pokemon_data(search_ia):
        st.session_state.ia_name = search_ia.capitalize()
        st.rerun()

auto_mode = st.sidebar.toggle("🚀 MODO AUTO-BATTLE")
velocidad = st.sidebar.slider("Delay Animación", 0.5, 3.0, 0.8)

# Datos de combate
ia_data = get_pokemon_data(st.session_state.ia_name)
rival_data = get_pokemon_data(st.session_state.rival_name)

# --- LÓGICA DE COMBATE ---
def combat_step(move_idx=None, is_switch=False, switch_name=None):
    if is_switch:
        st.session_state.ia_name = switch_name
        st.session_state.active_vfx = None
        st.session_state.historial.insert(0, f"🔄 IA cambió a {switch_name}")
    else:
        move = ia_data['moves'][move_idx]
        st.session_state.active_vfx = move['type']
        dmg = int(move['power'] * 0.2)
        st.session_state.hp_rival = max(0, st.session_state.hp_rival - dmg)
        st.session_state.historial.insert(0, f"🤖 {ia_data['name']} usó {move['name']}")
        st.session_state.rewards.append(st.session_state.rewards[-1] + dmg)

    # Respuesta Rival (Simulada para el Dashboard)
    if st.session_state.hp_rival > 0:
        riv_move = np.random.choice(rival_data['moves'])
        st.session_state.hp_ia = max(0, st.session_state.hp_ia - 10)
        st.session_state.historial.insert(0, f"💢 {rival_data['name']} usó {riv_move['name']}")
    else:
        st.session_state.historial.insert(0, "🏆 ¡VICTORIA!")
        st.session_state.hp_rival, st.session_state.hp_ia = 100, 100
        st.session_state.rival_name = np.random.choice(["Charizard", "Gyarados", "Dragonite", "Mewtwo", "Gengar"])
        st.balloons()

# --- RENDER ESCENA DE BATALLA ---
vfx_html = ""
if st.session_state.active_vfx == "Fire": vfx_html = '<div class="vfx-fire"></div>'
elif st.session_state.active_vfx == "Electric": vfx_html = '<div class="vfx-electric"></div>'
elif st.session_state.active_vfx: vfx_html = '<div class="vfx-hit">💥</div>'

battle_html = f"""
<div style="background-image: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 350px; position: relative; border: 4px solid #333; border-radius: 15px; overflow: hidden;">
    {vfx_html}
    <div style="position: absolute; top: 20px; right: 50px; text-align: center;">
        <div style="background: white; border: 2px solid #000; padding: 5px; border-radius: 5px; font-weight: bold; color: black; width: 140px; font-size: 12px;">
            {st.session_state.rival_name.upper()} {st.session_state.hp_rival}%
            <div style="background: #eee; height: 10px; border: 1px solid #000;"><div style="width: {st.session_state.hp_rival}%; background: #2ECC71; height: 100%;"></div></div>
        </div>
        <img src="{rival_data['sprite_front']}" width="120">
    </div>
    <div style="position: absolute; bottom: 20px; left: 50px; text-align: center;">
        <img src="{ia_data['sprite_back']}" width="180">
        <div style="background: white; border: 2px solid #000; padding: 5px; border-radius: 5px; font-weight: bold; color: black; width: 140px; font-size: 12px;">
            {st.session_state.ia_name.upper()} {st.session_state.hp_ia}%
            <div style="background: #eee; height: 10px; border: 1px solid #000;"><div style="width: {st.session_state.hp_ia}%; background: #2ECC71; height: 100%;"></div></div>
        </div>
    </div>
</div>
"""
st.components.v1.html(battle_html, height=360)

# El truco del refresco para las animaciones
if st.session_state.active_vfx:
    time.sleep(0.7)
    st.session_state.active_vfx = None
    st.rerun()

# --- CONTROLES ---
c1, c2, c3 = st.columns([1.5, 1, 1.2])

with c1:
    st.subheader("⚔️ Ataques")
    m_cols = st.columns(2)
    for i, m in enumerate(ia_data['moves']):
        btn_color = TYPE_COLORS.get(m['type'], "#666")
        st.markdown(f"<style>button[key='atk_{i}'] {{ background-color: {btn_color} !important; }}</style>", unsafe_allow_html=True)
        if m_cols[i%2].button(f"{m['name']}\n({m['type']})", key=f"atk_{i}", use_container_width=True):
            combat_step(move_idx=i)
            st.rerun()

with c2:
    st.subheader("🔄 Cambiar")
    # Sugerencias de cambio para la IA
    for p in ["Pikachu", "Gengar", "Lucario"]:
        if p != st.session_state.ia_name:
            if st.button(f"Ir a {p}", key=f"sw_{p}", use_container_width=True):
                combat_step(is_switch=True, switch_name=p)
                st.rerun()

with c3:
    st.subheader("📜 Log")
    with st.container(height=210, border=True):
        for l in st.session_state.historial: st.write(f"• {l}")

# --- GRÁFICOS ---
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.write("**Curva de Recompensa (RL)**")
    st.line_chart(st.session_state.rewards)
with g2:
    st.write("**Estadísticas Base (PokeAPI)**")
    st.bar_chart(pd.Series(ia_data['stats']))

if auto_mode:
    time.sleep(velocidad)
    combat_step(move_idx=np.random.randint(0, len(ia_data['moves'])))
    st.rerun()