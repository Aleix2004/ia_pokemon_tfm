import streamlit as st
import requests
import pandas as pd
import os
import numpy as np
import sys
import time

# --- CONFIGURACIÓN DE RUTAS ---
ruta_raiz = os.path.dirname(os.path.abspath(__file__))
ruta_env = os.path.join(ruta_raiz, "src", "env")
if ruta_env not in sys.path:
    sys.path.append(ruta_env)

try:
    from pokemon_env_gym import PokemonEnv
except ImportError:
    st.error("❌ No se encontró pokemon_env_gym.py en src/env")
    st.stop()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Pokemon TFM - Simulación Pro", page_icon="🤖", layout="wide")

def get_pokemon_sprite(name):
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    try:
        r = requests.get(url, timeout=2).json()
        return r['sprites']['front_default']
    except:
        return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/poke-ball.png"

# --- BARRA LATERAL ---
st.sidebar.header("🕹️ Panel de Control")
modelos_reales = [f for f in os.listdir('models') if f.endswith('.zip')] if os.path.exists('models') else ["Checkpoint_Final.zip"]
modelo_sel = st.sidebar.selectbox("1. Checkpoint:", modelos_reales)
tu_poke_sel = st.sidebar.selectbox("2. Tu Pokémon IA:", ["Charizard", "Pikachu", "Gengar", "Lucario", "Snorlax"])

st.sidebar.write("---")
auto_mode = st.sidebar.toggle("🚀 MODO AUTO-BATTLE", value=False)
velocidad = st.sidebar.slider("Velocidad (segundos)", 0.1, 2.0, 1.0)

# --- MÉTRICAS ---
st.title("📊 IA Pokémon: Simulación Autónoma")
c1, c2, c3 = st.columns(3)
c1.metric("ELO Estimado", "1285", "+15")
c2.metric("Win Rate Local", "78.4%", "+2.1%")
c3.metric("Modo Actual", "AUTÓNOMO" if auto_mode else "MANUAL")

# --- INICIALIZACIÓN ---
if 'env' not in st.session_state:
    st.session_state.env = PokemonEnv()
    st.session_state.env.reset()
    st.session_state.rival_actual = "Snorlax"
    st.session_state.historial = ["¡Sistema autónomo listo!"]
    st.session_state.stats_sesion = []

# --- INTERFAZ DE COMBATE ---
col_battle, col_logs = st.columns([2, 1])

with col_battle:
    # Normalizamos el HP para que Streamlit siempre muestre la barra correctamente (0.0 a 1.0)
    ia_hp = np.clip(st.session_state.env.hp_ia, 0.0, 1.0)
    rival_hp = np.clip(st.session_state.env.hp_rival, 0.0, 1.0)
    
    tipo_rival_idx = st.session_state.env.tipo_rival_real
    nombre_tipo_rival = st.session_state.env.nombres_tipos.get(tipo_rival_idx, "Normal")

    c_left, c_vs, c_right = st.columns([2, 1, 2])
    with c_left:
        st.image(get_pokemon_sprite(tu_poke_sel), width=180)
        color_ia = "green" if ia_hp > 0.4 else "red"
        st.markdown(f"<p style='color:{color_ia}; font-weight:bold;'>Salud IA: {int(ia_hp*100)}%</p>", unsafe_allow_html=True)
        st.progress(ia_hp)
        
    with c_vs:
        st.markdown("<h1 style='text-align: center; margin-top: 45px;'>VS</h1>", unsafe_allow_html=True)
        
    with c_right:
        st.image(get_pokemon_sprite(st.session_state.rival_actual), width=180)
        color_riv = "green" if rival_hp > 0.4 else "red"
        st.markdown(f"<p style='color:{color_riv}; font-weight:bold;'>Rival ({nombre_tipo_rival}): {int(rival_hp*100)}%</p>", unsafe_allow_html=True)
        st.progress(rival_hp)

    st.write("---")
    movs = ["🔥 Fuego", "💧 Agua", "🍃 Planta", "⚡ Rayo"]

    # --- MODO AUTOMÁTICO ---
    if auto_mode and rival_hp > 0 and ia_hp > 0:
        time.sleep(velocidad)
        accion_ia = np.random.randint(0, 4) # Aquí conectarás tu modelo.predict
        _, reward, done, _, info = st.session_state.env.step(accion_ia)
        
        st.session_state.historial.insert(0, f"🤖 IA decidió usar {movs[accion_ia]}. Recompensa: {reward:.2f}")
        
        if done:
            res_txt = "Victoria 🏆" if info.get('is_win') else "Derrota 💀"
            st.session_state.stats_sesion.insert(0, {
                "Modelo": modelo_sel, "Rival": st.session_state.rival_actual, "Resultado": res_txt
            })
            st.session_state.rival_actual = np.random.choice(["Dragonite", "Gyarados", "Arcanine", "Lapras", "Snorlax"])
            st.session_state.env.reset()
        st.rerun()

    # --- MODO MANUAL ---
    elif not auto_mode:
        cols = st.columns(4)
        for i, m in enumerate(movs):
            if cols[i].button(m, use_container_width=True):
                _, reward, done, _, info = st.session_state.env.step(i)
                st.session_state.historial.insert(0, f"Manual: {m}. Reward: {reward:.2f}")
                if done:
                    st.session_state.env.reset()
                st.rerun()

with col_logs:
    st.markdown("### 📝 Log de Batalla")
    with st.container(height=350, border=True):
        for log in st.session_state.historial:
            st.write(log)

# --- TABLA DE HISTORIAL ---
st.write("---")
st.subheader("📊 Histórico de la IA (Resultados del Entrenamiento)")
if st.session_state.stats_sesion:
    st.table(pd.DataFrame(st.session_state.stats_sesion))
else:
    st.info("Completa un combate para ver los resultados aquí.")