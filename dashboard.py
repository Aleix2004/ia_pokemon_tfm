import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import os
import numpy as np
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Pokémon TFM - Animaciones", layout="wide", page_icon="⚔️")

# --- FUNCIONES DE API (Resumidas para brevedad) ---
@st.cache_data
def get_pokemon_data(name):
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}", timeout=3).json()
        return {
            "name": r['name'].capitalize(),
            "sprite_front": r['sprites']['versions']['generation-v']['black-white']['animated']['front_default'],
            "sprite_back": r['sprites']['versions']['generation-v']['black-white']['animated']['back_default'],
            "types": [t['type']['name'].capitalize() for t in r['types']],
            "stats": {s['stat']['name']: s['base_stat'] for s in r['stats']},
            "moves": [{"name": m['move']['name'].replace("-", " ").title(), "type": "Normal", "power": 60} for m in r['moves'][:4]]
        }
    except: return None

# --- ESTADO DE SESIÓN ---
if 'ia_name' not in st.session_state:
    st.session_state.ia_name = "Raichu"
    st.session_state.rival_name = "Dragonite"
    st.session_state.hp_ia, st.session_state.hp_rival = 100, 100
    st.session_state.active_vfx = None
    st.session_state.rewards = [0]
    st.session_state.historial = ["¡Rama de animaciones activa!"]

# --- DATOS ---
ia_data = get_pokemon_data(st.session_state.ia_name)
rival_data = get_pokemon_data(st.session_state.rival_name)

# --- LÓGICA DE COMBATE ---
def combat_step(move_idx):
    move = ia_data['moves'][move_idx]
    st.session_state.active_vfx = move['name'] # Activamos el trigger de animación
    dmg = 15
    st.session_state.hp_rival = max(0, st.session_state.hp_rival - dmg)
    st.session_state.rewards.append(st.session_state.rewards[-1] + dmg)
    st.session_state.historial.insert(0, f"⚔️ {st.session_state.ia_name} usó {move['name']}!")

# --- DEFINICIÓN DE ANIMACIONES CSS ---
# Aquí es donde ocurre la magia de la Fase 1
animations_css = """
<style>
    /* 1. Movimiento Idle (Flotar) */
    @keyframes idle-float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* 2. Animación de Ataque (Embestida) */
    @keyframes attack-ia {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(60px, -30px) scale(1.1); }
        100% { transform: translate(0, 0) scale(1); }
    }

    /* 3. Animación de Daño (Vibración y Flash Rojo) */
    @keyframes damage-rival {
        0% { transform: translate(2px, 1px); filter: brightness(1); }
        20% { transform: translate(-3px, -2px); filter: brightness(2) sepia(1) hue-rotate(-50deg); }
        40% { transform: translate(3px, 2px); }
        60% { transform: translate(-3px, 1px); }
        100% { transform: translate(0, 0); filter: brightness(1); }
    }

    .anim-idle { animation: idle-float 3s ease-in-out infinite; }
    .anim-attack { animation: attack-ia 0.4s ease-in-out; }
    .anim-damage { animation: damage-rival 0.4s ease-in-out; }
</style>
"""

# --- RENDERIZADO DE LA ESCENA ---
# Determinamos qué clase aplicar basándonos en si hay un ataque en curso
ia_class = "anim-attack" if st.session_state.active_vfx else "anim-idle"
rival_class = "anim-damage" if st.session_state.active_vfx else "anim-idle"

vfx_banner = ""
if st.session_state.active_vfx:
    vfx_banner = f'''
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 100;">
        <div style="background: rgba(0,0,0,0.8); color: gold; padding: 10px 20px; border-radius: 10px; 
                    font-size: 28px; font-weight: bold; border: 2px solid gold; text-align: center;">
            {st.session_state.active_vfx.upper()}
        </div>
    </div>
    '''

battle_html = f'''
{animations_css}
<div style="position: relative; background-image: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); 
            background-size: cover; background-position: center; height: 350px; width: 100%;
            border-radius: 20px; border: 4px solid #222; overflow: hidden; font-family: sans-serif;">
    
    <div style="position: absolute; top: 30px; right: 60px; text-align: center;" class="{rival_class}">
        <div style="background: white; color: black; padding: 3px 10px; border-radius: 10px; 
                    font-weight: bold; border: 2px solid #333; font-size: 14px; margin-bottom: 5px;">
            {st.session_state.rival_name}: {st.session_state.hp_rival}%
        </div>
        <img src="{rival_data['sprite_front']}" width="130">
    </div>

    {vfx_banner}

    <div style="position: absolute; bottom: 20px; left: 60px; text-align: center;" class="{ia_class}">
        <img src="{ia_data['sprite_back']}" width="180">
        <div style="background: white; color: black; padding: 3px 10px; border-radius: 10px; 
                    font-weight: bold; border: 2px solid #333; font-size: 14px; margin-top: 5px;">
            {st.session_state.ia_name}: {st.session_state.hp_ia}%
        </div>
    </div>
</div>
'''

components.html(battle_html, height=360)

# Barras de vida
c1, _, c3 = st.columns([1, 1, 1])
with c1: st.progress(st.session_state.hp_ia / 100)
with c3: st.progress(st.session_state.hp_rival / 100)

# Control de tiempo para resetear la animación
if st.session_state.active_vfx:
    time.sleep(0.6)
    st.session_state.active_vfx = None
    st.rerun()

# --- BOTONES DE ATAQUE ---
st.write("---")
cols = st.columns(4)
for i, m in enumerate(ia_data['moves']):
    if cols[i].button(m['name'], use_container_width=True):
        combat_step(i)
        st.rerun()

st.subheader("📜 Log de Combate")
with st.container(height=120):
    for l in st.session_state.historial: st.write(f"• {l}")