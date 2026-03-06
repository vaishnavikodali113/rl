import numpy as np
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def load_sb3(path):
    data = np.load(path)
    return data["timesteps"], data["results"].mean(axis=1)

def load_tdmpc2(path):
    data = np.load(path)
    return data[:, 0], data[:, 1]

# PPO
ppo_path = "./logs/ppo_walker_mac/eval/evaluations.npz"
if os.path.exists(ppo_path):
    t, r = load_sb3(ppo_path)
    axes[0].plot(t, r, color="blue", label="PPO")
    axes[0].set_title("PPO — Walker Walk")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True)
    axes[0].legend()

# SAC
sac_path = "./logs/sac_cheetah_mac/eval/evaluations.npz"
if os.path.exists(sac_path):
    t, r = load_sb3(sac_path)
    axes[1].plot(t, r, color="orange", label="SAC")
    axes[1].set_title("SAC — Cheetah Run")
    axes[1].set_xlabel("Timesteps")
    axes[1].grid(True)
    axes[1].legend()

# TD-MPC2
tdmpc2_path = "./logs/tdmpc2_walker_mac/results.npy"
if os.path.exists(tdmpc2_path):
    t, r = load_tdmpc2(tdmpc2_path)
    axes[2].plot(t, r, color="green", label="TD-MPC2 (MLP)")
    axes[2].set_title("TD-MPC2 — Walker Walk")
    axes[2].set_xlabel("Timesteps")
    axes[2].grid(True)
    axes[2].legend()

plt.tight_layout()
plt.savefig("./logs/all_results.png", dpi=150)
plt.show()
print("Saved to ./logs/all_results.png")