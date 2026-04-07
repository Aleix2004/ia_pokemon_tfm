# 🐲 Proyecto TFM: IA Pokémon con Arquitectura Big Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SQLite](https://img.shields.io/badge/database-SQLite-green.svg)](https://www.sqlite.org/)

## 📝 1. Resumen e Introducción
Este proyecto desarrolla un ecosistema de **Inteligencia Artificial** capaz de combatir en entornos Pokémon, integrando un pipeline completo de **Big Data**. El sistema no solo toma decisiones en tiempo real mediante un modelo **PPO (Proximal Policy Optimization)**, sino que gestiona grandes volúmenes de datos mediante procesos ETL y almacenamiento relacional para su posterior análisis.

---

## 🎯 2. Objetivos del Proyecto
* **Ingesta Automatizada:** Extracción de datos desde PokeAPI.
* **Almacenamiento Escalable:** Implementación de una base de datos SQL para persistencia de stats y logs.
* **Entrenamiento de IA:** Aplicación de Reinforcement Learning para optimizar estrategias de combate.
* **Análisis de Datos:** Visualización de métricas de rendimiento y win-rate del modelo.

---

## 🛠️ 3. Herramientas y Tecnologías (Punto 4 PDF)
Siguiendo las orientaciones del proyecto final, se han utilizado:
* **Python:** Lenguaje principal para la lógica y el modelo de IA.
* **SQL (SQLite):** Motor de base de datos para el almacenamiento del Dataset.
* **Pandas:** Transformación y análisis de estructuras de datos.
* **Stable Baselines3:** Framework para la implementación del modelo PPO.
* **Streamlit:** Dashboard interactivo para la visualización de la Arena y la Capa SQL.
* **Docker:** Containerización para asegurar la portabilidad del proyecto.

---

## 📊 4. Metodología de Minería de Datos (CRISP-DM)
El proyecto sigue el ciclo de vida **CRISP-DM**:
1.  **Comprensión de Datos:** Análisis de los atributos de PokeAPI (HP, Attack, Defense, etc.).
2.  **Preparación (ETL):** Script `etl_process.py` que limpia y carga los datos en `pokemon_bigdata.db`.
3.  **Modelado:** Entrenamiento de la red neuronal en un entorno personalizado de OpenAI Gym.
4.  **Evaluación:** Registro automático en SQL de cada batalla para auditar el éxito de la IA.



---

## 🚀 5. Implementación Práctica
### Ingesta de Datos (ETL)
```bash
python etl_process.py

streamlit run dashboard.py