import numpy as np
import matplotlib.pyplot as plt
import os

# Configurar ruta al archivo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_NPZ = os.path.join(BASE_DIR, "logs", "evaluations.npz")

if os.path.exists(PATH_NPZ):
    # Cargar datos comprimidos de la IA
    data = np.load(PATH_NPZ)
    
    # 'results' son las recompensas obtenidas en las pruebas
    # 'timesteps' es el momento del entrenamiento en que se hicieron
    results = data['results']
    timesteps = data['timesteps']
    
    # Calculamos la media de las recompensas en cada evaluación
    mean_rewards = np.mean(results, axis=1)

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, marker='o', color='red', linestyle='-', linewidth=2)
    
    plt.title('Evolución de la Inteligencia (Evaluación de IA)', fontsize=14)
    plt.xlabel('Pasos de Entrenamiento (Timesteps)', fontsize=12)
    plt.ylabel('Recompensa Media', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar imagen para el TFM
    output_path = os.path.join(BASE_DIR, "grafica_tfm_ia.png")
    plt.savefig(output_path)
    
    print(f"✅ ¡Gráfica generada con éxito!")
    print(f"📍 Archivo guardado en: {output_path}")
    plt.show()
else:
    print(f"❌ No encontré el archivo en: {PATH_NPZ}")