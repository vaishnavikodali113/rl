import numpy as np
import matplotlib.pyplot as plt
import os

def load_evaluations(log_path):
    data = np.load(log_path)
    timesteps = data["timesteps"]
    results = data["results"].mean(axis=1)
    return timesteps, results

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PPO
ppo_path = "./logs/ppo_walker_mac/eval/evaluations.npz"
if os.path.exists(ppo_path):
    t, r = load_evaluations(ppo_path)
    axes[0].plot(t, r, label="PPO", color="blue")
    axes[0].set_title("PPO - Walker Walk")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend()
    axes[0].grid(True)

# SAC
sac_path = "./logs/sac_cheetah_mac/eval/evaluations.npz"
if os.path.exists(sac_path):
    t, r = load_evaluations(sac_path)
    axes[1].plot(t, r, label="SAC", color="orange")
    axes[1].set_title("SAC - Cheetah Run")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Mean Reward")
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.savefig("./logs/baseline_results.png", dpi=150)
plt.show()
print("Plot saved to ./logs/baseline_results.png")