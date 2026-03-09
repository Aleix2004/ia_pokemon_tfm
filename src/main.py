import gymnasium as gym
import wandb
import time
# Importamos la clase que acabas de crear
from env.pokemon_env import PokemonEnv

def main():
    # 1. Iniciar sesión en Weights & Biases
    # Esto creará un proyecto llamado 'pokemon-ia-tfm' en tu cuenta web
    wandb.init(
        project="pokemon-ia-tfm",
        name="test-inicial-semana-1",
        config={
            "env_name": "PokemonCustomEnv-v0",
            "type": "Random-Agent-Test"
        }
    )

    # 2. Instanciar tu entorno
    env = PokemonEnv()
    
    # 3. Ejecutar 5 combates de prueba
    for episodio in range(5):
        obs, _ = env.reset()
        terminated = False
        score = 0
        
        print(f"--- Iniciando Combate {episodio + 1} ---")
        
        while not terminated:
            # La IA elige un ataque al azar (0, 1, 2 o 3)
            action = env.action_space.sample() 
            
            # Ejecutamos el turno
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            # 4. ENVIAR DATOS A LA WEB (WandB)
            # Aquí es donde ocurre la magia de las gráficas
            wandb.log({
                "episodio": episodio,
                "reward_por_turno": reward,
                "vida_ia": obs[0],
                "vida_rival": obs[1],
                "puntuacion_total": score
            })
            
            time.sleep(0.1) # Para que no vaya demasiado rápido y podamos ver la terminal

        print(f"Combate finalizado. Puntuación: {score}")

    # 5. Cerrar el experimento
    wandb.finish()
    print("\n¡Prueba terminada con éxito! Revisa tu panel en wandb.ai")

if __name__ == "__main__":
    main()