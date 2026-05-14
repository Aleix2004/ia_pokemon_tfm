# TFM: IA Pokémon con Arquitectura Big Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable_Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/env-Gymnasium_0.29-green.svg)](https://gymnasium.farama.org/)
[![SQLite](https://img.shields.io/badge/database-SQLite-lightblue.svg)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://www.docker.com/)

Sistema completo de Inteligencia Artificial para combate Pokémon basado en **Reinforcement Learning** (PPO + Self-Play), con pipeline de Big Data ETL, base de datos relacional SQL y dashboard interactivo en Streamlit.

---

## Índice

1. [Arquitectura general](#arquitectura-general)
2. [Pipeline Big Data (ETL)](#pipeline-big-data-etl)
3. [Entorno de entrenamiento](#entorno-de-entrenamiento-gymnasium)
4. [Algoritmo PPO](#algoritmo-ppo)
5. [Función de recompensa](#función-de-recompensa)
6. [Infraestructura de entrenamiento](#infraestructura-de-entrenamiento)
7. [Self-Play](#self-play-alphazero-style)
8. [Dashboard](#dashboard-streamlit)
9. [Estructura del proyecto](#estructura-del-proyecto)
10. [Instalación y uso](#instalación-y-uso)
11. [Stack tecnológico](#stack-tecnológico)

---

## Arquitectura general

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUJO COMPLETO                           │
│                                                                 │
│  PokeAPI  ──ETL──►  pokemon_bigdata.db  ◄──  Dashboard SQL     │
│                            │                                    │
│                     pokemon_stats                               │
│                     battle_logs                                 │
│                     v_logs (turno a turno)                      │
│                                                                 │
│  PokemonEnv (Gymnasium)                                         │
│       │                                                         │
│       ▼                                                         │
│  SubprocVecEnv × 8  ──►  VecNormalize  ──►  PPO (MlpPolicy)   │
│       │                                                         │
│       ▼                                                         │
│  Self-Play: OpponentPool  ──►  5 generaciones entrenadas        │
│       │                                                         │
│       ▼                                                         │
│  BattleEngine  ──►  Streamlit Dashboard  (modo desafío + sim)  │
└─────────────────────────────────────────────────────────────────┘
```

La metodología seguida es **CRISP-DM**: comprensión del problema → preparación de datos (ETL) → modelado (PPO) → evaluación (métricas SQL + win-rate) → despliegue (dashboard + Docker).

---

## Pipeline Big Data (ETL)

### Fuente de datos

[PokeAPI](https://pokeapi.co/) — API REST pública que expone todos los atributos de los Pokémon en formato JSON.

### Proceso ETL (`etl_process.py`)

El pipeline sigue el patrón estándar **Extract → Transform → Load**:

**Extract:** llamadas HTTP a `https://pokeapi.co/api/v2/pokemon/{id}` para los 151 Pokémon de la región Kanto. Incluye retry con backoff exponencial (hasta 3 intentos, espera de 2 → 4 → 8 segundos) para tolerar fallos de red.

**Transform:** la función `_transform()` aplana el JSON anidado extrayendo únicamente los campos relevantes — HP, Attack, Defense, Sp.Attack, Sp.Defense, Speed, tipo primario y secundario — y los convierte en una tupla lista para inserción SQL.

**Load:** inserción en `pokemon_stats` mediante `INSERT OR REPLACE` (idempotente). Se hace `COMMIT` cada 10 registros para que el proceso sea recuperable ante interrupciones.

```bash
# Ejecutar el ETL completo (151 Pokémon)
python etl_process.py

# Test rápido con los primeros 20
python etl_process.py --limit 20
```

### Schema SQL (`database_manager.py`)

Tres tablas gestionadas por `init_db()` (idempotente, usa `CREATE TABLE IF NOT EXISTS`):

| Tabla | Propósito |
|-------|-----------|
| `pokemon_stats` | Dataset ETL — stats base de los 151 Pokémon |
| `battle_logs` | Historial de batallas: ganador y duración en turnos |
| `v_logs` | Log turno a turno: movimientos, efectividad de tipo, HP y reward |

### Análisis de datos (`check_data.py`)

Consultas SQL ejecutadas con Pandas para validar el dataset:

```bash
python check_data.py
# → Conteo de Pokémon por tipo primario
# → Top 5 atacantes físicos por base Attack
```

---

## Entorno de entrenamiento (Gymnasium)

**Fichero:** `src/env/pokemon_env.py`

Implementa la interfaz `gymnasium.Env` con contrato fijo:

| Parámetro | Valor |
|-----------|-------|
| `ENV_VERSION` | `pokemon_env_v1_obs28_act4` |
| Observation space | `Box(28,)` float32 |
| Action space | `Discrete(4)` |

### Observación (28 dimensiones)

| Rango | Contenido |
|-------|-----------|
| [0–1] | HP ratio de la IA y el rival (0.0 – 1.0) |
| [2–3] | Índices de tipo del Pokémon activo de la IA |
| [4–5] | Índices de tipo del Pokémon activo del rival |
| [6–11] | Etapas de stat de la IA (atk, def, sp_atk, sp_def, spd, acc) |
| [12–17] | Etapas de stat del rival |
| [18–19] | Flags de estado de la IA y el rival (burn, paralysis, etc.) |
| [20–23] | Tipos de los 4 movimientos disponibles |
| [24–27] | Potencias normalizadas de los 4 movimientos |

### Roster de entrenamiento

6 Pokémon con movesets competitivos: **Charizard, Blastoise, Venusaur, Pikachu, Garchomp, Alakazam**. Cada episodio enfrenta a uno de la IA contra uno del rival, elegidos aleatoriamente.

### Mecánicas del entorno de entrenamiento

El `PokemonEnv` implementa el subconjunto de mecánicas necesario para un aprendizaje estable:
- Cálculo de daño (fórmula oficial con efectividad de tipo)
- Stat stages (Swords Dance, Nasty Plot, etc.)
- Condiciones de status simplificadas

Las mecánicas completas (clima, piedras de entrada, parálisis, sueño, congelación) se implementan en `BattleEngine` para el dashboard, sin afectar al entrenamiento.

---

## Algoritmo PPO

**Fichero:** `src/train_ppo.py`

Se usa **Proximal Policy Optimization** de Stable Baselines3, con arquitectura `MlpPolicy` de dos cabezas independientes:

- **Policy head (π):** red neuronal `[256, 256]` → distribución sobre 4 acciones
- **Value head (V):** red neuronal `[256, 256]` → estimación del valor de estado

### Hiperparámetros de producción

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `learning_rate` | 3e-4 | Adam estándar para RL |
| `n_steps` | 2048 | Pasos por env por update |
| `batch_size` | 128 | Mini-batch para gradientes estables |
| `n_epochs` | 10 | Epochs PPO por update |
| `ent_coef` | 0.02 | Bonus de entropía para exploración |
| `clip_range` | 0.2 | Ratio de clipping PPO (ε) |
| `gamma` | 0.99 | Factor de descuento |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |

### Objetivo PPO (clipping)

```
L = min( r·Â,  clip(r, 1−ε, 1+ε)·Â )
```
donde `r = π(a|s) / π_old(a|s)` y `Â` es la ventaja estimada por GAE.

### Callbacks de entrenamiento

- `WinRateCallback` — registra la tasa de victoria cada N pasos en TensorBoard
- `VecNormCheckpointCallback` — guarda modelo + estadísticas de VecNormalize en cada checkpoint
- `SyncVecNormEvalCallback` — sincroniza las estadísticas de normalización entre el entorno de entrenamiento y el de evaluación

```bash
# Entrenamiento estándar
python -m src.train_ppo

# Con parámetros personalizados
python -m src.train_ppo --n_envs 16 --total_timesteps 1_000_000

# Continuar desde checkpoint
python -m src.train_ppo \
    --load_model   models/checkpoints/ppo_ckpt_200000_steps \
    --load_vecnorm models/checkpoints/ppo_ckpt_200000_steps_vecnorm.pkl
```

---

## Función de recompensa

**Fichero:** `src/reward_config.py`

15 componentes organizados en 3 niveles jerárquicos:

### Tier 1 — Señales terminales (dominantes)

| Componente | Valor | Descripción |
|------------|-------|-------------|
| `terminal_win` | +1.00 | Ganar la batalla |
| `terminal_loss` | −1.00 | Perder la batalla |
| `ko_bonus` | +0.22 | Noquear un Pokémon rival |
| `faint_penalty` | −0.22 | Ser noqueado |
| `anti_burst_penalty` | −0.20 | Descuento por ganar en ≤5 turnos |

### Tier 2 — Señales por turno (estrategia)

| Componente | Valor | Descripción |
|------------|-------|-------------|
| `damage_dealt_k` | 0.12 | Daño infligido normalizado |
| `survival_bonus` | 0.010 | Bonus por seguir vivo cada turno |
| `hp_lead_k` | 0.015 | Ventaja de HP proporcional |
| `consec_adv_k` | 0.003 | Dominancia consecutiva (compuesto) |
| `stall_penalty` | 0.015 | Penalización por stall real |

### Tier 3 — Señales posicionales (shaping)

| Componente | Valor | Descripción |
|------------|-------|-------------|
| `matchup_k` | 0.055 | Gradiente de ventaja de tipo |
| `bad_stay_matchup_k` | 0.035 | Penalización por quedarse en mal matchup |
| `temporal_risk_k` | 0.040 | Urgencia ante amenaza de KO |
| `momentum_k` | 0.020 | EMA del diferencial de daño |
| `move_quality_k` | 0.040 | Efectividad de tipo del movimiento elegido |
| `smart_switch_k` | 0.090 | Calidad del cambio de Pokémon |

El `RewardExplainer` permite imprimir un desglose completo del reward por paso para auditoría y depuración.

---

## Infraestructura de entrenamiento

**Fichero:** `src/train_ppo.py`

```
SubprocVecEnv(8)  →  VecNormalize  →  PPO
      │
      └── 8 procesos independientes (multiprocessing real, sin GIL)
              cada uno ejecuta su propio PokemonEnv
```

- **`SubprocVecEnv`:** 8 workers en procesos reales. Sin GIL de Python → paralelismo genuino. 8× más experiencia por unidad de tiempo de reloj.
- **`VecNormalize`:** normalización online (media y desviación estándar en tiempo real) de observaciones y recompensas. Estabiliza el gradiente cuando las magnitudes de los features son heterogéneas.
- **Evaluación paralela:** entorno de evaluación independiente (`DummyVecEnv`) con las estadísticas de normalización sincronizadas desde el entorno de entrenamiento.

---

## Self-Play (AlphaZero-style)

**Fichero:** `src/self_play.py`

El agente se entrena contra versiones anteriores de sí mismo, siguiendo la arquitectura de AlphaZero:

```
OpponentPool (buffer circular, max 10 versiones)
      │
      │  Sampling: 70% versión más reciente / 30% versiones antiguas
      ▼
VecEnv workers × 8  →  PPO update  →  nuevo checkpoint
      │
      ▼
Evaluación vs. todo el pool → actualización Elo
      │
      ▼
Añadir nueva versión al pool  →  siguiente generación
```

### Sistema Elo

- Rating inicial: 1500
- K-factor: 32
- Fórmula: `E_A = 1 / (1 + 10^((R_B − R_A) / 400))`

### Generaciones entrenadas

5 generaciones completadas: `self_play_gen_1.zip` → `self_play_gen_5.zip`, con 50.000 timesteps por generación y 8 entornos paralelos.

```bash
# Ejecutar self-play desde cero
python -m src.train_self_play

# Evaluar el modelo final
python -m src.evaluate_ia
```

---

## Dashboard Streamlit

**Fichero:** `dashboard.py`

Interfaz visual completa con dos modos:

**Modo Desafío:** el jugador humano elige su equipo y se enfrenta a la IA en tiempo real. El agente PPO decide cada turno mediante `model.predict(obs, deterministic=True)`. Se registra cada turno en `v_logs` y el resultado en `battle_logs`.

**Modo Simulación:** dos agentes PPO se enfrentan automáticamente. Útil para evaluar el comportamiento del modelo sin intervención humana.

Características visuales: sprites animados GIF, efectos de flash en ataques, panel de efectividad de tipo, selector de equipo con preview de stats, sección de análisis SQL con consultas en tiempo real sobre `pokemon_bigdata.db`.

```bash
streamlit run dashboard.py
```

---

## Estructura del proyecto

```
ia_pokemon_tfm/
│
├── dashboard.py               # Dashboard Streamlit (interfaz principal)
├── etl_process.py             # Pipeline ETL: PokeAPI → SQLite
├── database_manager.py        # Schema SQL e inicialización de tablas
├── check_data.py              # Consultas de validación del dataset
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Containerización
│
├── src/
│   ├── __init__.py
│   ├── main.py                # Punto de entrada del entrenamiento
│   ├── train_ppo.py           # Pipeline PPO con SubprocVecEnv + VecNormalize
│   ├── train_ia.py            # Pipeline simplificado (experimentos rápidos)
│   ├── train_self_play.py     # Bucle de self-play por generaciones
│   ├── self_play.py           # OpponentPool + EloRating
│   ├── evaluate_ia.py         # Evaluación del modelo final
│   ├── reward_config.py       # RewardWeights dataclass (15 componentes)
│   ├── model_compat.py        # Validación de compatibilidad de modelos
│   ├── battle_mechanics.py    # Status, clima, hazards
│   ├── battle_utils.py        # Utilidades de cálculo de daño y tipos
│   ├── pokemon_data.py        # Datos de Pokémon
│   ├── pokemon_forms.py       # Formas alternativas y shinies
│   ├── competitive_movesets.py # Movesets competitivos
│   ├── sprite_registry.py     # Registro de sprites
│   ├── sprites.py             # Carga y resolución de sprites
│   ├── type_colors.py         # Colores por tipo
│   ├── ai_advisor.py          # Consejero IA para el dashboard
│   ├── plot_evals.py          # Gráficas de evaluación
│   ├── plot_results.py        # Gráficas de resultados
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   └── pokemon_env.py     # PokemonEnv — entorno Gymnasium (obs28/act4)
│   │
│   └── game_engine/
│       ├── __init__.py
│       ├── battle_engine.py   # BattleEngine — simulador completo (dashboard)
│       └── obs_builder.py     # build_obs_28() — vector de observación compartido
│
├── models/                    # Modelos entrenados (excluidos del repo por tamaño)
│   ├── pokemon_ia_v5_avanzada.zip
│   ├── self_play_gen_1.zip … self_play_gen_5.zip
│   └── *.meta.json            # Metadatos de compatibilidad (env_version, obs_shape)
│
├── assets/sprites/            # Sprites GIF animados
├── logs/                      # TensorBoard logs
├── scripts/
│   ├── download_sprites.py
│   └── test_determinism.py
└── tests/
```

---

## Instalación y uso

### Requisitos

```bash
pip install -r requirements.txt
```

### 1. Inicializar la base de datos

```bash
python database_manager.py
```

### 2. Ejecutar el ETL

```bash
python etl_process.py          # 151 Pokémon de Kanto
python etl_process.py --limit 20  # test rápido
```

### 3. Verificar el dataset

```bash
python check_data.py
```

### 4. Entrenar el modelo

```bash
# Entrenamiento completo con 8 workers
python -m src.train_ppo

# Self-play por generaciones
python -m src.train_self_play
```

### 5. Evaluar el modelo

```bash
python -m src.evaluate_ia
```

### 6. Lanzar el dashboard

```bash
streamlit run dashboard.py
```

### Docker

```bash
docker build -t ia-pokemon .
docker run -p 8501:8501 ia-pokemon
```

---

## Stack tecnológico

| Categoría | Tecnología | Uso |
|-----------|------------|-----|
| Lenguaje | Python 3.10+ | Todo el proyecto |
| RL Framework | Stable Baselines3 | PPO, VecEnv, callbacks |
| Entorno RL | Gymnasium 0.29 | PokemonEnv |
| Deep Learning | PyTorch | Backend de SB3 |
| Big Data / ETL | Requests + SQLite | PokeAPI → BD relacional |
| Análisis datos | Pandas | Consultas SQL → DataFrame |
| Visualización | Streamlit + Plotly | Dashboard interactivo |
| Sprites | GIF animados | Assets visuales del dashboard |
| Contenedores | Docker | Despliegue reproducible |
| Tracking | TensorBoard | Métricas de entrenamiento |

---

## Autor

**Aleix Tallet** — Trabajo de Fin de Máster (TFM)
