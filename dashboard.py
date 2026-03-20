import streamlit as st
import streamlit.components.v1 as components
import requests
import os
import numpy as np
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pokémon IA - Battle Mode v5", layout="wide")

# --- FUNCIONES DE API (Sprites Animados Reales) ---
@st.cache_data
def get_pokemon_data(name):
    try:
        # Buscamos en la sección de 'black-white' -> 'animated' de PokéAPI
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}", timeout=5).json()
        gen5_sprites = r['sprites']['versions']['generation-v']['black-white']['animated']
        
        return {
            "name": r['name'].upper(),
            "sprite_front": gen5_sprites['front_default'], # Dragonite animado
            "sprite_back": gen5_sprites['back_default']    # Raichu animado de espalda
        }
    except: 
        # Fallback por si la API falla
        return {
            "name": name.upper(),
            "sprite_front": "https://play.pokemonshowdown.com/sprites/ani/dragonite.gif",
            "sprite_back": "https://play.pokemonshowdown.com/sprites/ani-back/raichu.gif"
        }

# --- ESTADO DE SESIÓN ---
if 'hp_ia' not in st.session_state:
    st.session_state.ia_name = "Raichu"
    st.session_state.rival_name = "Dragonite"
    st.session_state.hp_ia, st.session_state.hp_rival = 100, 100
    st.session_state.active_vfx = None
    st.session_state.historial = ["¡Módulo de combate listo! Esperando a la IA..."]

ia_data = get_pokemon_data(st.session_state.ia_name)
rival_data = get_pokemon_data(st.session_state.rival_name)

# --- LÓGICA DE COMUNICACIÓN ---
def llamar_a_la_ia():
    URL_API = "http://host.docker.internal:8000/predict"
    try:
        payload = {"ia_hp": int(st.session_state.hp_ia), "rival_hp": int(st.session_state.hp_rival)}
        response = requests.post(URL_API, json=payload, timeout=2)
        data = response.json()
        
        move_name = data['move_name']
        st.session_state.active_vfx = move_name
        
        # Daños dinámicos
        danos = {"Thunderbolt": 25, "Iron Tail": 20, "Quick Attack": 12, "Thunder Wave": 5}
        dmg = danos.get(move_name, 15)
        
        st.session_state.hp_rival = max(0, st.session_state.hp_rival - dmg)
        st.session_state.historial.insert(0, f"🧠 IA utilizó {move_name}")
    except:
        st.error("❌ Error: Verifica que server_ia.py esté corriendo en Windows.")

# --- CSS DE ANIMACIONES Y ESTILO POKÉMON ---
animations_css = """
<style>
    @import url('https://fonts.cdnfonts.com/css/pokemon-pixel-font');
    
    .battle-container {
        position: relative;
        background-image: url('https://play.pokemonshowdown.com/fx/bg-forest.png');
        background-size: cover;
        height: 450px;
        width: 100%;
        border-radius: 15px;
        border: 8px solid #333;
        overflow: hidden;
        image-rendering: pixelated;
    }

    /* Plataformas de suelo */
    .platform-ia { position: absolute; bottom: 40px; left: 60px; width: 250px; height: 60px; background: rgba(0,0,0,0.2); border-radius: 50%; z-index: 1; }
    .platform-rival { position: absolute; top: 180px; right: 80px; width: 200px; height: 50px; background: rgba(0,0,0,0.2); border-radius: 50%; z-index: 1; }

    /* Animaciones */
    .anim-idle { animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }

    .anim-attack { animation: attack 0.5s ease-in-out; }
    @keyframes attack { 0% { transform: translate(0,0); } 50% { transform: translate(80px, -40px); } 100% { transform: translate(0,0); } }

    .anim-damage { animation: damage 0.5s ease-in-out; }
    @keyframes damage { 0%, 100% { filter: brightness(1); } 50% { filter: brightness(10) sepia(1); transform: translateX(10px); } }

    /* Barras de Vida Estilo GameBoy */
    .hp-card {
        background: #f8f8f8;
        border: 3px solid #333;
        border-radius: 0 0 0 15px;
        padding: 8px;
        width: 220px;
        font-family: 'monospace';
        box-shadow: 4px 4px 0px #888;
    }
    .hp-bar-bg { background: #444; width: 100%; height: 10px; border: 1px solid #000; margin-top: 5px; }
    .hp-bar-fill { height: 100%; transition: width 0.5s ease-in-out; }
</style>
"""

# --- RENDERIZADO DEL COMBATE ---
st.title("🎮 IA Pokémon TFM - Battle Interface")

ia_class = "anim-attack" if st.session_state.active_vfx else "anim-idle"
rival_class = "anim-damage" if st.session_state.active_vfx else "anim-idle"

# Banner de movimiento
vfx_html = f'''<div style="position: absolute; top: 40%; left: 50%; transform: translate(-50%, -50%); z-index: 100; background: black; color: white; padding: 10px 20px; border: 4px solid gold; font-family: monospace; font-size: 24px;">{st.session_state.active_vfx.upper()}</div>''' if st.session_state.active_vfx else ""

battle_html = f'''
{animations_css}
<div class="battle-container">
    <div class="platform-rival"></div>
    <div style="position: absolute; top: 40px; right: 50px; z-index: 2;" class="{rival_class}">
        <div class="hp-card">
            <div style="display: flex; justify-content: space-between;"><b>{ia_data['name']}</b> <span>Lv. 100</span></div>
            <div class="hp-bar-bg"><div class="hp-bar-fill" style="width: {st.session_state.hp_rival}%; background: #00ff00;"></div></div>
        </div>
        <img src="{rival_data['sprite_front']}" width="180">
    </div>

    {vfx_html}

    <div class="platform-ia"></div>
    <div style="position: absolute; bottom: 30px; left: 40px; z-index: 2;" class="{ia_class}">
        <img src="{ia_data['sprite_back']}" width="280">
        <div class="hp-card" style="margin-top: -20px; border-radius: 15px 0 0 0;">
            <div style="display: flex; justify-content: space-between;"><b>{ia_data['name']} (IA)</b> <span>Lv. 100</span></div>
            <div class="hp-bar-bg"><div class="hp-bar-fill" style="width: {st.session_state.hp_ia}%; background: #00ff00;"></div></div>
            <div style="text-align: right; font-size: 10px;">{st.session_state.hp_ia}/100</div>
        </div>
    </div>
</div>
'''

components.html(battle_html, height=460)

# --- BOTONES ---
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("🔴 EJECUTAR TURNO IA", use_container_width=True, type="primary"):
        llamar_a_la_ia()
        st.rerun()
with c2:
    if st.button("🔄 REINICIAR", use_container_width=True):
        st.session_state.hp_ia, st.session_state.hp_rival = 100, 100
        st.rerun()

if st.session_state.active_vfx:
    time.sleep(0.8)
    st.session_state.active_vfx = None
    st.rerun()

st.subheader("📝 Historial de Combate")
for l in st.session_state.historial[:5]: st.text(f"• {l}")