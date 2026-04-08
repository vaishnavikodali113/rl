import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from env_setup import make_env

def evaluate_policy(model, env, n_episodes=20, device='cpu'):
    """
    Run model in env for n_episodes using MPPI planning.
    Returns: mean_reward, std_reward
    """
    from planning.mppi import MPPI
    planner = MPPI(model, env.action_space.shape[0], horizon=5, n_samples=512)
    episode_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0] # handle gym/gymnasium reset returns
        if obs.ndim == 1: obs = obs[np.newaxis, :] # ensure batched
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        z = model.encoder(obs_tensor)
        total_reward = 0.0
        done = False
        while not done:
            action = planner.plan(z, device).squeeze(0).cpu().numpy()
            obs, reward, done, *_ = env.step([action] if isinstance(env, DummyVecEnv) else action)
            
            # Handle vectorized vs non-vectorized returns
            r = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            d = done[0] if isinstance(done, (list, np.ndarray)) else done
            o = obs[0] if isinstance(obs, (list, np.ndarray)) and obs.ndim > 1 else obs
            
            obs_tensor = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            z = model.encoder(obs_tensor)
            total_reward += float(r)
            done = bool(d)
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

def run_all_evaluations(algorithms: dict, env_name="walker", task="walk", n_episodes=20, device="cpu"):
    """
    algorithms: {"ppo_walker": model_obj, "tdmpc2_s5": model_obj, ...}
    Saves results to logs/comparison_table.csv
    """
    env = make_env(env_name, task)
    results = {}
    for name, model in algorithms.items():
        model.to(device)
        model.eval()
        mean, std = evaluate_policy(model, env, n_episodes, device=device)
        results[name] = {"mean_reward": mean, "std_reward": std}
        print(f"{name}: {mean:.1f} ± {std:.1f}")
    
    # Save to CSV
    import csv
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/comparison_table.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "mean_reward", "std_reward"])
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({"algorithm": name, **r})
    return results
