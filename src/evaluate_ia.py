import os

import numpy as np

try:
    from src.env.pokemon_env import PokemonEnv
    from src.model_compat import check_model_compatibility, require_compatible_model
except ImportError:
    from env.pokemon_env import PokemonEnv
    from model_compat import check_model_compatibility, require_compatible_model


def run_eval(model, opponent_mode="random", opponent_model=None, episodes=100):
    env = PokemonEnv()
    env.set_opponent(mode=opponent_mode, model=opponent_model, random_baseline_chance=0.5)

    wins = 0
    rewards = []
    lengths = []
    dealt = []
    received = []
    ko_counts = []
    stalled_turns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
        wins += 1 if info.get("is_win") else 0
        rewards.append(info.get("episode_reward", 0.0))
        lengths.append(info.get("episode_length", 0))
        dealt.append(info.get("damage_dealt", 0.0))
        received.append(info.get("damage_received", 0.0))
        ko_counts.append(info.get("ko_count", 0))
        stalled_turns.append(info.get("stalled_turns", 0))

    return {
        "win_rate": wins / episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths)),
        "avg_damage_dealt": float(np.mean(dealt)),
        "avg_damage_received": float(np.mean(received)),
        "avg_ko_count": float(np.mean(ko_counts)),
        "avg_stalled_turns": float(np.mean(stalled_turns)),
    }


def _find_compatible_snapshot():
    history_dir = "models/self_play_history"
    if not os.path.isdir(history_dir):
        return None
    candidates = []
    for name in os.listdir(history_dir):
        if name.endswith(".zip"):
            base = os.path.join(history_dir, name[:-4])
            compat = check_model_compatibility(base)
            if compat.is_valid:
                candidates.append(base)
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def evaluate():
    candidate_paths = [
        "models/canonical_ppo_v1",
        "models/best_self_play/model_final_v4",
        "models/best_model_s3/best_model",
    ]
    model_path = next((path for path in candidate_paths if os.path.exists(path + ".zip")), None)
    if model_path is None:
        print("No trained PPO model was found for evaluation.")
        return

    compat = check_model_compatibility(model_path)
    if not compat.is_valid:
        raise RuntimeError(
            f"Primary model '{model_path}' is LEGACY - INCOMPATIBLE ({compat.reason}). "
            "Use a model trained with canonical env contract (obs28/action4)."
        )

    print(f"Loading model from {model_path}.zip")
    model = require_compatible_model(model_path)
    random_metrics = run_eval(model, opponent_mode="random", episodes=100)
    greedy_metrics = run_eval(model, opponent_mode="greedy", episodes=100)

    snapshot_path = _find_compatible_snapshot()
    if snapshot_path is None:
        print("No compatible PPO snapshot found in models/self_play_history. Using primary model as mirror opponent.")
        snapshot_model = require_compatible_model(model_path)
    else:
        print(f"Using compatible snapshot opponent: {snapshot_path}.zip")
        snapshot_model = require_compatible_model(snapshot_path)
    snapshot_metrics = run_eval(model, opponent_mode="model", opponent_model=snapshot_model, episodes=100)

    print("\nEvaluation complete")
    print("Vs random baseline")
    print(f"  Win rate: {random_metrics['win_rate']:.2%} | Avg reward: {random_metrics['avg_reward']:.3f}")
    print(f"  Avg battle length: {random_metrics['avg_length']:.2f} | Avg KOs: {random_metrics['avg_ko_count']:.3f}")
    print(f"  Avg damage dealt: {random_metrics['avg_damage_dealt']:.3f} | Avg damage received: {random_metrics['avg_damage_received']:.3f}")
    print(f"  Avg stalled turns: {random_metrics['avg_stalled_turns']:.2f}")

    print("\nVs greedy baseline")
    print(f"  Win rate: {greedy_metrics['win_rate']:.2%} | Avg reward: {greedy_metrics['avg_reward']:.3f}")
    print(f"  Avg battle length: {greedy_metrics['avg_length']:.2f} | Avg KOs: {greedy_metrics['avg_ko_count']:.3f}")
    print(f"  Avg damage dealt: {greedy_metrics['avg_damage_dealt']:.3f} | Avg damage received: {greedy_metrics['avg_damage_received']:.3f}")
    print(f"  Avg stalled turns: {greedy_metrics['avg_stalled_turns']:.2f}")

    print("\nVs frozen mirror policy")
    print(f"  Win rate: {snapshot_metrics['win_rate']:.2%} | Avg reward: {snapshot_metrics['avg_reward']:.3f}")
    print(f"  Avg battle length: {snapshot_metrics['avg_length']:.2f} | Avg KOs: {snapshot_metrics['avg_ko_count']:.3f}")
    print(f"  Avg damage dealt: {snapshot_metrics['avg_damage_dealt']:.3f} | Avg damage received: {snapshot_metrics['avg_damage_received']:.3f}")
    print(f"  Avg stalled turns: {snapshot_metrics['avg_stalled_turns']:.2f}")


if __name__ == "__main__":
    evaluate()
