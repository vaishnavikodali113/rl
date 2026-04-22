import numpy as np
import torch

from env_setup import make_env

def _reset_model_hidden(model, batch_size: int, device: torch.device | str) -> None:
    if hasattr(model.dynamics, "reset_hidden"):
        model.dynamics.reset_hidden(batch_size=batch_size, device=device)


def _planner_from_config(model, action_dim: int, planner_config: dict | None = None):
    from planning.mppi import MPPI

    config = dict(planner_config or getattr(model, "planner_config", {}) or {})
    return MPPI(
        model,
        action_dim,
        horizon=config.get("horizon", config.get("plan_horizon", 5)),
        n_samples=config.get("n_samples", config.get("plan_samples", 512)),
        temperature=config.get("temperature", config.get("plan_temperature", 0.5)),
        gamma=config.get("gamma", 0.99),
        noise_scale=config.get("noise_scale", 0.25),
    )


def evaluate_policy(model, env, n_episodes=20, device='cpu', planner_config: dict | None = None):
    """
    Run model in env for n_episodes using MPPI planning.
    Returns: mean_reward, std_reward
    """
    device = torch.device(device)
    planner = None if hasattr(model, "act") else _planner_from_config(
        model,
        env.action_space.shape[0],
        planner_config=planner_config,
    )
    episode_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        obs_value = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0.0
        done = False
        episode_step = 0
        if not hasattr(model, "act"):
            _reset_model_hidden(model, batch_size=1, device=device)
        while not done:
            with torch.no_grad():
                if hasattr(model, "act"):
                    action_tensor = model.act(
                        torch.as_tensor(obs_value, dtype=torch.float32),
                        t0=episode_step == 0,
                        eval_mode=True,
                    ).unsqueeze(0)
                else:
                    obs_tensor = torch.as_tensor(obs_value, dtype=torch.float32, device=device).unsqueeze(0)
                    z = model.encoder(obs_tensor)
                    env_hidden = (
                        model.dynamics.snapshot_hidden()
                        if hasattr(model.dynamics, "snapshot_hidden")
                        else None
                    )
                    action_tensor = planner.plan(z, device)
                    if hasattr(model.dynamics, "restore_hidden"):
                        model.dynamics.restore_hidden(env_hidden)
            action = action_tensor.squeeze(0).cpu().numpy()
            if not hasattr(model, "act"):
                with torch.no_grad():
                    if hasattr(model.dynamics, "restore_hidden"):
                        model.dynamics.restore_hidden(env_hidden)
                    _ = model.dynamics(z, action_tensor)
                    if hasattr(model.dynamics, "snapshot_hidden"):
                        env_hidden = model.dynamics.snapshot_hidden()
            obs, reward, done, *_ = env.step(action)
            obs_value = obs[0] if isinstance(obs, tuple) else obs
            reward_value = reward[0] if isinstance(reward, (tuple, list, np.ndarray)) else reward
            total_reward += float(reward_value)
            episode_step += 1
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

def run_all_evaluations(algorithms: dict, env_name="walker", task="walk", n_episodes=20, device="cpu", planner_configs: dict[str, dict] | None = None):
    """
    algorithms: {"ppo_walker": model_obj, "tdmpc2_s5": model_obj, ...}
    Saves results to logs/comparison_table.csv
    """
    env = make_env(env_name, task)
    results = {}
    for name, model in algorithms.items():
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        mean, std = evaluate_policy(model, env, n_episodes, device=device, planner_config=(planner_configs or {}).get(name))
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
