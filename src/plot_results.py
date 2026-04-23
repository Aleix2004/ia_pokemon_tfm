import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 1. Configurar rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def plot_log_files():
    # Buscamos todos los archivos .monitor.csv en la carpeta logs
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.monitor.csv"))
    
    if not log_files:
        print(f"❌ No se encontraron archivos .monitor.csv en {LOGS_DIR}")
        return

    plt.figure(figsize=(10, 6))

    for file in log_files:
        # Leer el CSV saltando las primeras líneas de comentarios
        df = pd.read_csv(file, skiprows=1)
        
        # 'r' es la recompensa (reward) y 'l' es la longitud del episodio (length)
        # Calculamos la media móvil para suavizar la gráfica
        df['rolling_reward'] = df['r'].rolling(window=50).mean()
        
        plt.plot(df['rolling_reward'], label=os.path.basename(file))

    plt.title('Progreso del Entrenamiento - Pokémon IA')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Media (Suavizada)')
    plt.legend()
    plt.grid(True)
    
    # Guardar la imagen para tu memoria del TFM
    grafica_path = os.path.join(BASE_DIR, "curva_aprendizaje.png")
    plt.savefig(grafica_path)
    print(f"✅ Gráfica guardada como: {grafica_path}")
    plt.show()

if __name__ == "__main__":
    plot_log_files()
    