import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_metrics_jsonl(path):
    """Load training metrics from a JSONL artifact file."""
    timesteps, rewards = [], []
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # SB3 logs reward under different keys; TD-MPC2 logs it directly
            r = d["metrics"].get("eval/mean_reward",
                d["metrics"].get("rollout/ep_rew_mean", 
                d["metrics"].get("rollout/recent_episode_return", None)))
            if r is not None:
                timesteps.append(d["timesteps"])
                rewards.append(r)
    return np.array(timesteps), np.array(rewards)

def plot_reward_curves(configs: dict, save_path="logs/fig1_reward_curves.png"):
    """
    configs: {
        "PPO (Walker)":        "artifacts/ppo_walker/metrics.jsonl",
        "SAC (Cheetah)":       "artifacts/sac_cheetah/metrics.jsonl",
        "TD-MPC2 MLP":         "artifacts/tdmpc2_walker_mlp/metrics.jsonl",
        "TD-MPC2 S5":          "artifacts/tdmpc2_walker_s5/metrics.jsonl",
        ...
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ["#95a5a6", "#e67e22", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, path) in enumerate(configs.items()):
        t, r = load_metrics_jsonl(path)
        if t is None:
            continue
        ax.plot(t, r, label=label, color=colors[i % len(colors)], linewidth=2)
        
    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Training Performance: All Algorithms", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

def plot_planning_stability(results: dict, save_path="logs/fig3_planning_stability.png"):
    """
    results: {
        "MLP H=5":   final_reward,
        "MLP H=10":  final_reward,
        "S5 H=5":    final_reward,
        "S5 H=10":   final_reward,
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [("MLP", "#e74c3c"), ("S5", "#2ecc71")]
    
    x = np.array([5, 10])
    bar_width = 0.35
    for i, (name, color) in enumerate(groups):
        heights = [results.get(f"{name} H=5", 0), results.get(f"{name} H=10", 0)]
        ax.bar(x + i * bar_width, heights, bar_width, label=name, color=color, alpha=0.85)
        
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(["Horizon 5", "Horizon 10"], fontsize=12)
    ax.set_ylabel("Final Mean Reward", fontsize=13)
    ax.set_title("Planning Stability: MLP vs SSM at Different Horizons", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

def plot_sample_efficiency(configs: dict, checkpoints=(50_000, 100_000, 200_000), save_path="logs/fig4_sample_efficiency.png"):
    """
    Shows reward at specific training checkpoints for each algorithm.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(checkpoints))
    bar_width = 0.2
    
    valid_configs = {}
    for k, v in configs.items():
        t, r = load_metrics_jsonl(v)
        if t is not None and len(t) > 0:
            valid_configs[k] = (t, r)
            
    for i, (label, (t, r)) in enumerate(valid_configs.items()):
        # Interpolate reward at each checkpoint
        heights = [float(np.interp(ck, t, r)) for ck in checkpoints]
        ax.bar(x + i * bar_width, heights, bar_width, label=label, alpha=0.85)
        
    ax.set_xticks(x + bar_width * (len(valid_configs) - 1) / 2)
    ax.set_xticklabels([f"{ck // 1000}k steps" for ck in checkpoints], fontsize=11)
    ax.set_ylabel("Mean Reward at Checkpoint", fontsize=13)
    ax.set_title("Sample Efficiency Comparison", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
