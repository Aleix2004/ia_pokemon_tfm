import streamlit as st
import requests
import pandas as pd
import os
import time
import numpy as np
import random
from stable_baselines3 import PPO
from src.env.pokemon_env import PokemonEnv 

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA Pokémon TFM - Ultimate", layout="wide", page_icon="🐲")

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
        'env': PokemonEnv()
    })

def combat_step(action_idx):
    curr_ia = st.session_state.team_ia[st.session_state.active_ia]
    curr_riv = st.session_state.team_rival[st.session_state.active_rival]
    old_hp_riv = st.session_state.env.hp_rival
    old_hp_ia = st.session_state.env.hp_ia

    # Pasar el nombre del ataque al env para el log
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
        else: st.session_state.battle_finished = True; st.session_state.resultado = "¡GANASTE! 🎉"

    if st.session_state.env.hp_ia <= 0:
        st.session_state.team_ia[st.session_state.active_ia]['debilitado'] = True
        siguiente = next((i for i, p in enumerate(st.session_state.team_ia) if not p['debilitado']), None)
        if siguiente is not None:
            st.session_state.active_ia = siguiente; st.session_state.env.hp_ia = 1.0
            st.session_state.historial.insert(0, f"💀 **TUYA KO**: Sale {st.session_state.team_ia[siguiente]['name']}")
        else: st.session_state.battle_finished = True; st.session_state.resultado = "PERDISTE 💀"

# --- 4. PANTALLA SELECCIÓN ---
if not st.session_state.game_started and not st.session_state.battle_finished:
    st.title("🧪 Configuración TFM")
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

# --- 5. SIDEBAR (Buscador Recursivo) ---
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
    if curr_ia['item']:
        st.divider(); st.subheader("🎒 Objeto")
        st.image(curr_ia['item']['sprite'], width=40); st.write(curr_ia['item']['name'])
    st.divider(); st.subheader("📊 Stats"); st.table(pd.Series(curr_ia['stats']))

# --- 6. ARENA Y MARCADOR --- (Copia y sustituye este bloque)
v_ia, v_riv = sum(1 for p in st.session_state.team_ia if not p['debilitado']), sum(1 for p in st.session_state.team_rival if not p['debilitado'])
st.markdown(f"<h2 style='text-align: center;'>🏟️ Marcador: IA {v_ia}/6  vs  RIVAL {v_riv}/6</h2>", unsafe_allow_html=True)

curr_riv = st.session_state.team_rival[st.session_state.active_rival]
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
hp_ia, hp_riv = int(st.session_state.env.hp_ia*100), int(st.session_state.env.hp_rival*100)

st.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 280px; border-radius: 15px; position: relative; border: 2px solid #333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    
    <div style="position: absolute; top: 20px; right: 40px; width: 180px; background: rgba(0,0,0,0.7); padding: 8px; border-radius: 10px; border: 1px solid #555; z-index: 10;">
        <div style="display: flex; justify-content: space-between; color: white; margin-bottom: 4px;">
            <span style="font-weight: bold; font-size: 14px;">{curr_riv['name']}</span>
            <span style="font-size: 14px; color: #4CAF50;">{hp_riv}%</span>
        </div>
        <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="width: {hp_riv}%; background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%;"></div>
        </div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 45px; right: 10px; filter: drop-shadow(2px 4px 6px black);" width="80">
    </div>

    <div style="position: absolute; bottom: 20px; left: 40px; width: 180px; background: rgba(0,0,0,0.7); padding: 8px; border-radius: 10px; border: 1px solid #555; z-index: 10;">
        <div style="display: flex; justify-content: space-between; color: white; margin-bottom: 4px;">
            <span style="font-weight: bold; font-size: 14px;">{curr_ia['name']}</span>
            <span style="font-size: 14px; color: #4CAF50;">{hp_ia}%</span>
        </div>
        <div style="width: 100%; background: #333; height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="width: {hp_ia}%; background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%;"></div>
        </div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 50px; left: 10px; filter: drop-shadow(2px 4px 6px black);" width="110">
    </div>
</div>
''')

# --- 7. ACCIONES | LOGS | CAMBIOS ---
col_acc, col_log, col_sw = st.columns([1, 2, 1])

with col_acc:
    st.subheader("⚔️ Ataques")
    if not auto:
        for i, m in enumerate(curr_ia['moves']):
            if st.button(m['name'], key=f"at{i}", use_container_width=True):
                combat_step(i); st.rerun()

with col_log:
    st.subheader("📜 Registro")
    log_placeholder = st.empty()
    with log_placeholder.container(height=280, border=True):
        for e in st.session_state.historial:
            if "⚔️" in e: st.markdown(f"<span style='color: #00d4ff;'>{e}</span>", unsafe_allow_html=True)
            elif "🔴" in e: st.markdown(f"<span style='color: #ff4b4b;'>{e}</span>", unsafe_allow_html=True)
            elif "💥" in e: st.success(e)
            elif "💀" in e: st.error(e)
            else: st.write(e)

with col_sw:
    st.subheader("🔄 Cambios")
    for idx, p in enumerate(st.session_state.team_ia):
        dead = p.get('debilitado', False); active = (idx == st.session_state.active_ia)
        lbl = f"💀 {p['name']}" if dead else p['name']
        if st.button(lbl, key=f"sw{idx}", use_container_width=True, disabled=dead or active):
            st.session_state.active_ia = idx; st.session_state.env.hp_ia = 1.0
            st.session_state.historial.insert(0, f"🔄 Sale {p['name']}"); st.rerun()

# --- 8. EJECUCIÓN MODO AUTO ---
if auto and not st.session_state.battle_finished:
    if sel_model:
        model_path = os.path.join(MODELS_DIR, sel_model)
        model = PPO.load(model_path)
        action, _ = model.predict(st.session_state.env._get_obs())
        combat_step(int(action))
        time.sleep(vel)
        st.rerun() 

if st.session_state.battle_finished:
    st.success(st.session_state.resultado)
    if st.button("Reiniciar"): st.session_state.clear(); st.rerun()