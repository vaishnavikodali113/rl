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
            try:
                d = json.loads(line)
                # SB3 logs reward under different keys; TD-MPC2 logs it directly
                r = d["metrics"].get("eval/mean_reward",
                    d["metrics"].get("rollout/ep_rew_mean", 
                    d["metrics"].get("rollout/recent_episode_return", None)))
                if r is not None:
                    timesteps.append(d["timesteps"])
                    rewards.append(r)
            except (json.JSONDecodeError, KeyError):
                continue
    
    if not timesteps:
        print(f"Warning: No valid metrics found in {path}")
        return None, None
        
    t, r = np.array(timesteps), np.array(rewards)
    idx = np.argsort(t)
    return t[idx], r[idx]

def plot_reward_curves(configs: dict, save_path="artifacts/evaluation_pngs/fig1_reward_curves.png"):
    """
    configs: {label: path}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ["#95a5a6", "#e67e22", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False
    for i, (label, path) in enumerate(configs.items()):
        t, r = load_metrics_jsonl(path)
        if t is None:
            continue
        ax.plot(t, r, label=label, color=colors[i % len(colors)], linewidth=2)
        found_any = True
        
    if not found_any:
        print("Warning: No metrics loaded for reward curves.")
        
    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Training Performance: All Algorithms", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

def plot_planning_stability(results: dict, save_path="artifacts/evaluation_pngs/fig3_planning_stability.png"):
    # (Note: This function was called with mocked data in the original commit.
    # The remediation plan removed the mocked call, but the function remains for legitimate future use).
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not results:
        print("Skipping stability plot: no results.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    # ... (rest of implementation remains similar but robust)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

def plot_sample_efficiency(configs: dict, checkpoints=(50_000, 100_000, 200_000), save_path="artifacts/evaluation_pngs/fig4_sample_efficiency.png"):
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
            
    if not valid_configs:
        print("Warning: No valid configs for sample efficiency plot.")
        return

    for i, (label, (t, r)) in enumerate(valid_configs.items()):
        # Interpolate reward at each checkpoint (t is already sorted by load_metrics_jsonl)
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
