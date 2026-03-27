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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Pokémon TFM - Final Build", layout="wide", page_icon="🐲")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- 2. MOTOR POKEAPI CON CACHÉ ---
@st.cache_data
def get_pokemon_data(name_or_id, is_shiny=None):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{str(name_or_id).lower().strip()}"
        r = requests.get(url, timeout=5).json()
        if is_shiny is None: is_shiny = random.random() < 0.10
        
        # Sprites animados de Gen V
        animated = r['sprites']['versions']['generation-v']['black-white']['animated']
        s_key = 'front_shiny' if is_shiny else 'front_default'
        sb_key = 'back_shiny' if is_shiny else 'back_default'
        img_f = animated[s_key] if animated[s_key] else r['sprites'][s_key]
        img_b = animated[sb_key] if animated[sb_key] else r['sprites'][sb_key]

        stats = {s['stat']['name'].replace("-", " ").title(): s['base_stat'] for s in r['stats']}
        species_url = r['species']['url']
        species_data = requests.get(species_url).json()
        megas = [v['pokemon']['name'] for v in species_data['varieties'] if "mega" in v['pokemon']['name']]

        return {
            "name": r['name'].capitalize(),
            "species_name": species_data['name'],
            "sprite_front": img_f, "sprite_back": img_b,
            "is_shiny": is_shiny, "available_megas": megas,
            "stats": stats, "debilitado": False,
            "moves": [{"name": m['move']['name'].replace("-", " ").capitalize()} for m in r['moves'][:4]]
        }
    except: return None

# --- 3. GESTIÓN DE ESTADO DE SESIÓN ---
if 'game_started' not in st.session_state:
    st.session_state.update({
        'game_started': False, 'battle_finished': False, 'resultado': "",
        'active_ia': 0, 'active_rival': 0,
        'damage_data': {}, 'mvp_data': {}, 'historial': [],
        'env': PokemonEnv()
    })

def combat_step(action_idx):
    # Lógica Mega-Evolución Rival Aleatoria
    curr_riv = st.session_state.team_rival[st.session_state.active_rival]
    if curr_riv['available_megas'] and random.random() < 0.10:
        m_data = get_pokemon_data(curr_riv['available_megas'][0], is_shiny=curr_riv['is_shiny'])
        if m_data:
            st.session_state.team_rival[st.session_state.active_rival] = m_data
            st.session_state.historial.insert(0, f"🌟 ¡RIVAL MEGA-EVOLUCIONÓ a {m_data['name']}!")

    old_hp_riv = st.session_state.env.hp_rival
    obs, reward, terminated, truncated, info = st.session_state.env.step(action_idx)
    damage_pct = max(0, (old_hp_riv - st.session_state.env.hp_rival) * 100)
    
    pkmn = st.session_state.team_ia[st.session_state.active_ia]
    key = pkmn['species_name'].capitalize()
    if key in st.session_state.damage_data: st.session_state.damage_data[key] += damage_pct

    # Registro en el Log
    if action_idx == 3:
        st.session_state.historial.insert(0, f"✨ {pkmn['name']} usó movimiento de ESTADO")
    elif action_idx < 4:
        move = pkmn['moves'][action_idx]['name']
        st.session_state.historial.insert(0, f"⚔️ {pkmn['name']} usó {move} -{damage_pct:.1f}%")

    # Gestión de KOs Rival
    if st.session_state.env.hp_rival <= 0:
        st.session_state.mvp_data[key] = st.session_state.mvp_data.get(key, 0) + 1
        st.session_state.team_rival[st.session_state.active_rival]['debilitado'] = True
        if st.session_state.active_rival < 5:
            st.session_state.active_rival += 1
            st.session_state.env.hp_rival = 1.0
            st.session_state.historial.insert(0, f"💥 RIVAL KO: Entra {st.session_state.team_rival[st.session_state.active_rival]['name']}")
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "GANASTE 🎉"

    # Gestión de KOs IA
    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]['debilitado'] = True
        siguiente = next((i for i, p in enumerate(st.session_state.team_ia) if not p['debilitado']), None)
        if siguiente is not None:
            st.session_state.active_ia = siguiente
            st.session_state.env.hp_ia = 1.0
            st.session_state.historial.insert(0, f"💀 TUYA KO: Sale {st.session_state.team_ia[siguiente]['name']}")
        else:
            st.session_state.battle_finished = True
            st.session_state.resultado = "PERDISTE 💀"

# --- 4. PANTALLA SELECCIÓN (EQUIPOS VISUALES) ---
if not st.session_state.game_started and not st.session_state.battle_finished:
    st.title("🧪 Configuración de Combate TFM")
    
    def build_team_ui(defaults, prefix, color):
        team = []
        with st.container(border=True):
            st.markdown(f"<h3 style='text-align: center; color: {color};'>EQUIPO {prefix.upper()}</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, name in enumerate(defaults):
                with cols[i%3]:
                    n = st.text_input(f"Slot {i+1}", value=name, key=f"{prefix}_{i}")
                    d = get_pokemon_data(n)
                    if d:
                        with st.container(border=True):
                            st.image(d['sprite_front'], width=70)
                            st.caption(f"<div style='text-align: center;'><b>{d['name']}</b></div>", unsafe_allow_html=True)
                        team.append(d)
        return team

    c1, c2 = st.columns(2)
    with c1: t_ia = build_team_ui(["Charizard", "Gengar", "Lucario", "Metagross", "Rayquaza", "Sceptile"], "ia", "#00d4ff")
    with c2: t_riv = build_team_ui(["Mewtwo", "Arceus", "Garchomp", "Kyogre", "Tyranitar", "Zoroark"], "rival", "#ff4b4b")
    
    st.divider()
    if st.button("🔥 INICIAR SIMULACIÓN", use_container_width=True, type="primary"):
        if len(t_ia) == 6 and len(t_riv) == 6:
            st.session_state.update({
                'team_ia': t_ia, 'team_rival': t_riv, 'active_ia': 0, 'active_rival': 0,
                'damage_data': {p['species_name'].capitalize(): 0 for p in t_ia},
                'mvp_data': {p['species_name'].capitalize(): 0 for p in t_ia},
                'game_started': True, 'historial': ["¡Combate iniciado!"]
            })
            st.session_state.env.reset(); st.rerun()
    st.stop()

# --- 5. RESUMEN FINAL ---
if st.session_state.battle_finished:
    st.header(f"🏁 Análisis Final: {st.session_state.resultado}")
    c1, c2 = st.columns(2)
    with c1: st.subheader("KOs por Pokémon"); st.bar_chart(pd.Series(st.session_state.mvp_data))
    with c2: st.subheader("Daño Total %"); st.bar_chart(pd.Series(st.session_state.damage_data))
    if st.button("Volver al Menú"): st.session_state.update({'game_started': False, 'battle_finished': False}); st.rerun()
    st.stop()

# --- 6. BARRA LATERAL (CONTROL Y LOG) ---
with st.sidebar:
    st.header("⚙️ Control IA")
    all_zips = [os.path.relpath(os.path.join(r, f), MODELS_DIR) for r, _, fs in os.walk(MODELS_DIR) for f in fs if f.endswith(".zip")]
    sel_model = st.selectbox("Modelo PPO:", all_zips)
    auto = st.toggle("Modo Automático", value=False)
    vel = st.slider("Velocidad", 0.1, 2.0, 0.5)
    
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    st.divider()
    st.subheader(f"📊 Stats: {curr_ia['name']}")
    st.table(pd.Series(curr_ia['stats']))
    st.divider()
    st.subheader("📜 Log en Vivo")
    log_area = st.empty()

def refresh_log():
    with log_area.container(border=True):
        for e in st.session_state.historial[:6]:
            if "⚔️" in e: st.write(f" `{e}`")
            elif "🌟" in e or "💥" in e: st.success(e)
            elif "💀" in e: st.error(e)
            else: st.caption(e)

# --- 7. MARCADOR Y ESCENARIO ---
v_ia = sum(1 for p in st.session_state.team_ia if not p['debilitado'])
v_riv = sum(1 for p in st.session_state.team_rival if not p['debilitado'])

st.markdown(f"<h2 style='text-align: center;'>🏟️ Marcador: IA {v_ia}/6  —  RIVAL {v_riv}/6</h2>", unsafe_allow_html=True)

curr_riv = st.session_state.team_rival[st.session_state.active_rival]
hp_ia, hp_riv = int(st.session_state.env.hp_ia*100), int(st.session_state.env.hp_rival*100)
c_ia = "#4CAF50" if hp_ia > 50 else "#FF9800" if hp_ia > 20 else "#F44336"
c_riv = "#4CAF50" if hp_riv > 50 else "#FF9800" if hp_riv > 20 else "#F44336"

st.components.v1.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 260px; border-radius: 15px; position: relative; border: 2px solid #333; font-family: sans-serif;">
    <div style="position: absolute; top: 20px; right: 40px; width: 150px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 8px;">
        <small><b>{curr_riv['name']}</b> {hp_riv}%</small>
        <div style="width: 100%; background: #ddd; height: 8px; border-radius: 4px;"><div style="width: {hp_riv}%; background: {c_riv}; height: 100%; border-radius: 4px;"></div></div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 35px; right: 0px;" width="80">
    </div>
    <div style="position: absolute; bottom: 20px; left: 40px; width: 150px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 8px;">
        <small><b>{curr_ia['name']}</b> {hp_ia}%</small>
        <div style="width: 100%; background: #ddd; height: 8px; border-radius: 4px;"><div style="width: {hp_ia}%; background: {c_ia}; height: 100%; border-radius: 4px;"></div></div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 40px; left: 0px;" width="110">
    </div>
</div>
''', height=280)

# --- 8. CONTROLES Y CAMBIO BLOQUEADO ---
c_atks, c_sw = st.columns([2, 1])
with c_atks:
    st.subheader("⚔️ Acciones")
    if not auto:
        m_cols = st.columns(4)
        for i, m in enumerate(curr_ia['moves'] + [{"name": "Estado ✨"}]):
            if i < 4 and m_cols[i].button(m['name'], key=f"at{i}", use_container_width=True):
                combat_step(i); st.rerun()
        if curr_ia['available_megas'] and st.button("🧬 MEGA-EVOLUCIONAR", use_container_width=True):
            m_d = get_pokemon_data(curr_ia['available_megas'][0], is_shiny=curr_ia['is_shiny'])
            st.session_state.team_ia[st.session_state.active_ia] = m_d
            st.session_state.historial.insert(0, "🌟 ¡Has Mega-Evolucionado!"); st.rerun()
    else:
        if sel_model:
            model = PPO.load(os.path.join(MODELS_DIR, sel_model))
            action, _ = model.predict(st.session_state.env._get_obs(), deterministic=True)
            combat_step(int(action)); refresh_log(); time.sleep(vel); st.rerun()

with c_sw:
    st.subheader("🔄 Cambiar")
    for idx, p in enumerate(st.session_state.team_ia):
        muerto = p.get('debilitado', False)
        actual = (idx == st.session_state.active_ia)
        label = f"💀 {p['name']}" if muerto else p['name']
        
        # Bloqueo real: disabled=True si está muerto o ya está en pista
        if st.button(label, key=f"sw{idx}", use_container_width=True, disabled=muerto or actual):
            st.session_state.active_ia = idx
            st.session_state.env.hp_ia = 1.0 
            st.session_state.historial.insert(0, f"🔄 Sale {p['name']}"); st.rerun()

refresh_log()