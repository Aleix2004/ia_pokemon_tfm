import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import os
import time
import numpy as np
import random
from stable_baselines3 import PPO
from src.env.pokemon_env import PokemonEnv 

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA Pokémon TFM - Master Control", layout="wide", page_icon="🐲")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- 2. MOTOR POKEAPI MEJORADO ---
@st.cache_data
def get_pokemon_data(name_or_id, is_shiny=None):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        r = requests.get(url, timeout=5).json()
        if is_shiny is None: is_shiny = random.random() < 0.10
        
        # Sprites
        animated = r['sprites']['versions']['generation-v']['black-white']['animated']
        s_key = 'front_shiny' if is_shiny else 'front_default'
        sb_key = 'back_shiny' if is_shiny else 'back_default'
        img_f = animated[s_key] if animated[s_key] else r['sprites'][s_key]
        img_b = animated[sb_key] if animated[sb_key] else r['sprites'][sb_key]

        # Stats Base
        stats = {s['stat']['name']: s['base_stat'] for s in r['stats']}

        # Mega Evoluciones
        species_data = requests.get(r['species']['url']).json()
        megas = [v['pokemon']['name'] for v in species_data['varieties'] if "mega" in v['pokemon']['name']]

        return {
            "name": r['name'].capitalize(),
            "species_name": species_data['name'],
            "sprite_front": img_f, "sprite_back": img_b,
            "is_shiny": is_shiny, "available_megas": megas,
            "stats": stats,
            "moves": [{"name": m['move']['name'].replace("-", " ").capitalize()} for m in r['moves'][:4]]
        }
    except: return None

# --- 3. GESTIÓN DE ESTADO ---
if 'game_started' not in st.session_state:
    st.session_state.update({
        'game_started': False, 'battle_finished': False, 'resultado': "",
        'active_ia': 0, 'active_rival': 0,
        'damage_data': {}, 'mvp_data': {}, 'historial': [],
        'env': PokemonEnv()
    })

def combat_step(action_idx):
    # Lógica de Mega-Evolución Rival Aleatoria
    curr_riv = st.session_state.team_rival[st.session_state.active_rival]
    if curr_riv['available_megas'] and random.random() < 0.05: # 5% cada turno
        mega_name = curr_riv['available_megas'][0]
        mega_data = get_pokemon_data(mega_name, is_shiny=curr_riv['is_shiny'])
        if mega_data:
            st.session_state.team_rival[st.session_state.active_rival] = mega_data
            st.session_state.historial.insert(0, f"🌟 ¡El RIVAL ha Mega-Evolucionado a {mega_data['name']}!")

    old_hp_riv = st.session_state.env.hp_rival
    obs, reward, terminated, truncated, info = st.session_state.env.step(action_idx)
    
    damage = max(0, (old_hp_riv - st.session_state.env.hp_rival) * 100)
    pkmn = st.session_state.team_ia[st.session_state.active_ia]
    key = pkmn['species_name'].capitalize()
    
    if key in st.session_state.damage_data:
        st.session_state.damage_data[key] += damage

    # Historial detallado
    if action_idx == 3:
        st.session_state.historial.insert(0, f"✨ {pkmn['name']} usó movimiento de ESTADO")
    elif action_idx < 4:
        move = pkmn['moves'][action_idx]['name']
        st.session_state.historial.insert(0, f"⚔️ {pkmn['name']} usó {move} (-{damage:.1f}%)")

    # Gestión de KOs
    if st.session_state.env.hp_rival <= 0:
        st.session_state.mvp_data[key] = st.session_state.mvp_data.get(key, 0) + 1
        if st.session_state.active_rival < 5:
            st.session_state.active_rival += 1
            st.session_state.env.hp_rival = 1.0
            st.session_state.historial.insert(0, f"💥 RIVAL KO: Entra {st.session_state.team_rival[st.session_state.active_rival]['name']}")
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "GANASTE 🎉"

    if st.session_state.env.hp_ia <= 0:
        if st.session_state.active_ia < 5:
            st.session_state.active_ia += 1
            st.session_state.env.hp_ia = 1.0
            st.session_state.historial.insert(0, f"💀 TUYA KO: Sale {st.session_state.team_ia[st.session_state.active_ia]['name']}")
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "PERDISTE 💀"

# --- 4. UI: PANTALLA SELECCIÓN ---
if not st.session_state.game_started and not st.session_state.battle_finished:
    st.title("🧪 Configuración de Combate")
    def build_team(defaults, key):
        cols = st.columns(3); team = []
        for i, name in enumerate(defaults):
            with cols[i%3]:
                n = st.text_input(f"Slot {i+1}", value=name, key=f"{key}_{i}")
                d = get_pokemon_data(n)
                if d:
                    with st.container(border=True):
                        st.image(d['sprite_front'], width=60)
                        st.caption(f"**{d['name']}**")
                    team.append(d)
        return team
    c1, c2 = st.columns(2)
    with c1: t_ia = build_team(["Charizard", "Gengar", "Lucario", "Metagross", "Rayquaza", "Sceptile"], "ia")
    with c2: t_riv = build_team(["Mewtwo", "Arceus", "Garchomp", "Kyogre", "Tyranitar", "Zoroark"], "riv")
    if st.button("🔥 INICIAR SIMULACIÓN", use_container_width=True):
        st.session_state.update({
            'team_ia': t_ia, 'team_rival': t_riv, 'active_ia': 0, 'active_rival': 0,
            'damage_data': {p['species_name'].capitalize(): 0 for p in t_ia},
            'mvp_data': {p['species_name'].capitalize(): 0 for p in t_ia},
            'game_started': True, 'historial': []
        })
        st.session_state.env.reset(); st.rerun()
    st.stop()

# --- 5. UI: RESUMEN FINAL ---
if st.session_state.battle_finished:
    st.header(f"🏁 Análisis de Rendimiento Final")
    st.subheader(f"Resultado: {st.session_state.resultado}")
    col_mvp, col_dmg = st.columns(2)
    with col_mvp:
        st.markdown("🥇 **KOs por Pokémon (MVP)**")
        st.bar_chart(pd.Series(st.session_state.mvp_data), color="#FFD700")
    with col_dmg:
        st.markdown("🔥 **Daño Total Acumulado (%)**")
        st.bar_chart(pd.Series(st.session_state.damage_data), color="#FF5252")
    if st.button("Reiniciar", use_container_width=True):
        st.session_state.update({'battle_finished': False, 'game_started': False}); st.rerun()
    st.stop()

# --- 6. BARRA LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Panel de Control")
all_zips = []
for root, dirs, files in os.walk(MODELS_DIR):
    for f in files:
        if f.endswith(".zip"): all_zips.append(os.path.relpath(os.path.join(root, f), MODELS_DIR))

selected_model_path = st.sidebar.selectbox("Modelo IA:", all_zips)
auto_mode = st.sidebar.toggle("Modo Automático", value=False)
speed = st.sidebar.slider("Delay", 0.1, 2.0, 0.6)

# Sección de Stats Base
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
curr_riv = st.session_state.team_rival[st.session_state.active_rival]

st.sidebar.divider()
st.sidebar.subheader(f"📊 Stats: {curr_ia['name']}")
df_stats = pd.DataFrame(curr_ia['stats'].items(), columns=['Stat', 'Value'])
st.sidebar.dataframe(df_stats, hide_index=True, use_container_width=True)

st.sidebar.markdown("""<div style="font-size: 0.8em; color: gray;">🔥 > 🍃 | 💧 > 🔥 | 🍃 > 💧 | ⚡ > 💧</div>""", unsafe_allow_html=True)

# --- 7. ARENA ---
hp_ia = int(st.session_state.env.hp_ia * 100)
hp_riv = int(st.session_state.env.hp_rival * 100)
c_ia = "#4CAF50" if hp_ia > 50 else "#FF9800" if hp_ia > 20 else "#F44336"
c_riv = "#4CAF50" if hp_riv > 50 else "#FF9800" if hp_riv > 20 else "#F44336"

st.components.v1.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 260px; border-radius: 15px; border: 2px solid #444; position: relative;">
    <div style="position: absolute; top: 20px; right: 40px; width: 160px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 8px;">
        <small>{curr_riv['name']} ({hp_riv}%)</small>
        <div style="width: 100%; background: #eee; height: 8px; border-radius: 4px;">
            <div style="width: {hp_riv}%; background: {c_riv}; height: 100%; border-radius: 4px;"></div>
        </div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 30px; right: 0px;" width="80">
    </div>
    <div style="position: absolute; bottom: 20px; left: 40px; width: 160px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 8px;">
        <small>{curr_ia['name']} ({hp_ia}%)</small>
        <div style="width: 100%; background: #eee; height: 8px; border-radius: 4px;">
            <div style="width: {hp_ia}%; background: {c_ia}; height: 100%; border-radius: 4px;"></div>
        </div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 40px; left: 0px;" width="110">
    </div>
</div>
''', height=280)

# --- 8. ACCIONES Y CAMBIOS ---
col_moves, col_switch = st.columns([2, 1])

with col_moves:
    st.subheader("⚔️ Ataques y Estado")
    if not auto_mode:
        m_cols = st.columns(4)
        moves_labels = [m['name'] for m in curr_ia['moves']] + ["Estado ✨"]
        for i, label in enumerate(moves_labels[:4]):
            if m_cols[i].button(label, key=f"atk_{i}", use_container_width=True):
                combat_step(i); st.rerun()
        
        if curr_ia['available_megas'] and st.button(f"🧬 MEGA-EVOLUCIONAR A {curr_ia['available_megas'][0].upper()}", use_container_width=True):
            mega = get_pokemon_data(curr_ia['available_megas'][0], is_shiny=curr_ia['is_shiny'])
            if mega:
                st.session_state.team_ia[st.session_state.active_ia] = mega
                st.session_state.historial.insert(0, f"🌟 ¡Mega-Evolución activada!")
                st.rerun()
    else:
        if selected_model_path:
            model = PPO.load(os.path.join(MODELS_DIR, selected_model_path))
            action, _ = model.predict(st.session_state.env._get_obs(), deterministic=True)
            combat_step(int(action))
            time.sleep(speed); st.rerun()

with col_switch:
    st.subheader("🔄 Cambiar")
    for idx, p in enumerate(st.session_state.team_ia):
        if idx != st.session_state.active_ia:
            if st.button(f"{p['name']}", key=f"sw_{idx}", use_container_width=True):
                st.session_state.active_ia = idx
                st.session_state.env.tipo_ia = random.randint(0, 2) # Simula cambio de tipo en env
                st.session_state.historial.insert(0, f"🔄 Cambio a {p['name']}")
                st.rerun()

st.divider()
st.subheader("📜 Historial")
for e in st.session_state.historial[:5]: st.write(e)