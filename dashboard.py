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
st.set_page_config(page_title="Control de IA Pokémon", layout="wide", page_icon="🐲")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

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

        species_data = requests.get(r['species']['url']).json()
        megas = [v['pokemon']['name'] for v in species_data['varieties'] if "mega" in v['pokemon']['name']]

        return {
            "name": r['name'].capitalize(),
            "species_name": species_data['name'],
            "sprite_front": img_f, "sprite_back": img_b,
            "is_shiny": is_shiny, "available_megas": megas,
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
    old_hp_riv = st.session_state.env.hp_rival
    obs, reward, terminated, truncated, info = st.session_state.env.step(action_idx)
    
    damage = max(0, (old_hp_riv - st.session_state.env.hp_rival) * 100)
    pkmn = st.session_state.team_ia[st.session_state.active_ia]
    key = pkmn['species_name'].capitalize()
    
    # 📊 Registrar Daño
    if key in st.session_state.damage_data:
        st.session_state.damage_data[key] += damage

    # ⚔️ Registro de Movimientos
    if action_idx < 4:
        move = pkmn['moves'][action_idx]['name']
        st.session_state.historial.insert(0, f"⚔️ {pkmn['name']} usó {move} (-{damage:.1f}%)")

    # 💀 Gestión de KOs y MVPs
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
    if st.button("🔥 INICIAR", use_container_width=True):
        st.session_state.update({
            'team_ia': t_ia, 'team_rival': t_riv, 'active_ia': 0, 'active_rival': 0,
            'damage_data': {p['species_name'].capitalize(): 0 for p in t_ia},
            'mvp_data': {p['species_name'].capitalize(): 0 for p in t_ia},
            'game_started': True, 'historial': []
        })
        st.session_state.env.reset(); st.rerun()
    st.stop()

# --- 5. UI: RESUMEN FINAL (ESTADÍSTICAS REUPERADAS) ---
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

    if st.button("Reiniciar Simulación", use_container_width=True):
        st.session_state.update({'battle_finished': False, 'game_started': False})
        st.rerun()
    st.stop()

# --- 6. ARENA DE COMBATE ---
curr_ia = st.session_state.team_ia[st.session_state.active_ia]
curr_riv = st.session_state.team_rival[st.session_state.active_rival]

# Barra Lateral
st.sidebar.header("⚙️ Panel de Control")
zips = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
selected_model = st.sidebar.selectbox("Modelo PPO:", zips)
auto_mode = st.sidebar.toggle("Modo Automático", value=False)
speed = st.sidebar.slider("Delay (s)", 0.1, 2.0, 0.6)

if curr_ia['available_megas'] and st.sidebar.button(f"🧬 Mega-Evolucionar"):
    mega = get_pokemon_data(curr_ia['available_megas'][0], is_shiny=curr_ia['is_shiny'])
    if mega:
        mega['species_name'] = curr_ia['species_name']
        st.session_state.team_ia[st.session_state.active_ia] = mega; st.rerun()

# Escenario
hp_ia = int(st.session_state.env.hp_ia * 100)
hp_riv = int(st.session_state.env.hp_rival * 100)
c_ia = "#4CAF50" if hp_ia > 50 else "#FF9800" if hp_ia > 20 else "#F44336"
c_riv = "#4CAF50" if hp_riv > 50 else "#FF9800" if hp_riv > 20 else "#F44336"

st.components.v1.html(f'''
<div style="background: url('https://play.pokemonshowdown.com/fx/bg-forest.png'); background-size: cover; height: 260px; border-radius: 15px; border: 2px solid #444; position: relative; font-family: sans-serif;">
    <div style="position: absolute; top: 20px; right: 40px; width: 180px; background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px;">
        <div style="font-weight: bold; color: #333;">{curr_riv['name']} ({hp_riv}%)</div>
        <div style="width: 100%; background: #eee; height: 10px; border-radius: 5px; margin-top: 5px;">
            <div style="width: {hp_riv}%; background: {c_riv}; height: 100%; border-radius: 5px;"></div>
        </div>
        <img src="{curr_riv['sprite_front']}" style="position: absolute; top: 40px; right: 10px;" width="90">
    </div>
    <div style="position: absolute; bottom: 20px; left: 40px; width: 180px; background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px;">
        <div style="font-weight: bold; color: #333;">{curr_ia['name']} ({hp_ia}%)</div>
        <div style="width: 100%; background: #eee; height: 10px; border-radius: 5px; margin-top: 5px;">
            <div style="width: {hp_ia}%; background: {c_ia}; height: 100%; border-radius: 5px;"></div>
        </div>
        <img src="{curr_ia['sprite_back']}" style="position: absolute; bottom: 50px; left: 10px;" width="120">
    </div>
</div>
''', height=280)

# Lógica IA / Manual
if not auto_mode:
    st.subheader(f"🎮 Turno de {curr_ia['name']}")
    cols = st.columns(4)
    for i, move in enumerate(curr_ia['moves']):
        if cols[i].button(move['name'], key=f"b_{i}", use_container_width=True):
            combat_step(i); st.rerun()
else:
    if selected_model:
        model = PPO.load(os.path.join(MODELS_DIR, selected_model))
        obs = st.session_state.env._get_obs()
        
        # 🛡️ ADAPTADOR DE SHAPE (4 vs 5)
        target_shape = model.observation_space.shape[0]
        if len(obs) < target_shape:
            obs = np.append(obs, [0.0] * (target_shape - len(obs)))
        elif len(obs) > target_shape:
            obs = obs[:target_shape]
        
        action, _ = model.predict(obs)
        combat_step(int(action))
    else: combat_step(random.randint(0, 3))
    time.sleep(speed); st.rerun()

# Registro en Vivo
st.divider()
st.subheader("📜 Registro de Combate")
for e in st.session_state.historial[:5]: st.write(e)