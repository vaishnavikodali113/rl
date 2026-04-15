import numpy as np
import matplotlib.pyplot as plt

def compute_horizon_error_curve(model, test_sequences, max_horizon=10, device='cpu'):
    """
    test_sequences: list of (obs_seq, act_seq) tensors, each of length max_horizon
    Returns: array of shape [max_horizon] with MSE at each prediction step
    """
    import torch
    import torch.nn.functional as F
    errors = np.zeros(max_horizon)
    n = len(test_sequences)
    
    for obs_seq, act_seq in test_sequences:
        obs_seq = obs_seq.to(device)   # [T+1, obs_dim]
        act_seq = act_seq.to(device)   # [T, act_dim]
        horizon = min(max_horizon, act_seq.shape[0], obs_seq.shape[0] - 1)
        if horizon <= 0:
            continue
        z0 = model.encoder(obs_seq[0].unsqueeze(0)).detach()
        
        if hasattr(model.dynamics, 'reset_hidden'):
            model.dynamics.reset_hidden(1)

        z = z0
        for t in range(horizon):
            z = model.dynamics(z, act_seq[t].unsqueeze(0))
            z_true = model.encoder(obs_seq[t + 1].unsqueeze(0)).detach()
            mse = F.mse_loss(z, z_true).item()
            errors[t] += mse / n
            
    return errors

def plot_rollout_errors(error_curves: dict, save_path="logs/fig2_rollout_error.png"):
    """
    error_curves: {"MLP": array[10], "S4": array[10], "S5": array[10], "Mamba": array[10]}
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"MLP": "#e74c3c", "S4": "#3498db", "S5": "#2ecc71", "Mamba": "#9b59b6"}
    
    for name, errors in error_curves.items():
        horizons = np.arange(1, len(errors) + 1)
        ax.plot(horizons, errors, label=name, color=colors.get(name, "gray"),
                linewidth=2, marker='o', markersize=5)
                
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=13)
    ax.set_ylabel("Latent MSE", fontsize=13)
    ax.set_title("Multi-Step Rollout Prediction Error: MLP vs SSM Dynamics", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
