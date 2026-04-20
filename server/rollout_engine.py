import numpy as np
import torch

from env_setup import make_env
from planning.mppi import MPPI


class ModelAgent:
    """Wraps a single model + env pair for uniform step() interface."""
    def __init__(self, model_dict: dict, device: str, mppi_horizon: int, mppi_samples: int):
        self.label = model_dict["label"]
        self.run_name = model_dict.get("run_name", self.label)
        self.display_name = model_dict.get("display_name", self.label)
        self.algorithm_name = model_dict.get("algorithm_name", self.label)
        self.env_title = model_dict.get("env_title", self.label)
        self.behavior_text = model_dict.get("behavior_text", self.env_title)
        self.algo_type = model_dict["algo_type"]
        self.model = model_dict["model"]
        self.device = device
        
        self.env = make_env(
            model_dict["env_name"],
            model_dict["task"],
            vectorized=False,
            render_mode="rgb_array",
        )
        self.obs, _ = self.env.reset()
        
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False

        # MPPI planner only for TD-MPC2 models
        if self.algo_type == "tdmpc":
            act_dim = self.env.action_space.shape[0]
            self.planner = MPPI(
                self.model,
                act_dim,
                horizon=mppi_horizon,
                n_samples=mppi_samples,
                action_low=torch.as_tensor(
                    self.env.action_space.low,
                    dtype=torch.float32,
                    device=device,
                ),
                action_high=torch.as_tensor(
                    self.env.action_space.high,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            # Encode initial observation
            obs_t = torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0)
            self.z = self.model.encoder(obs_t)
        else:
            self.planner = None
            self.z = None

    def step(self):
        """Advance environment by one step. Returns (frame, step_metrics)."""
        if self.done:
            self.obs, _ = self.env.reset()
            self.total_reward = 0.0
            self.step_count = 0
            self.done = False
            
            if self.algo_type == "tdmpc":
                obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.z = self.model.encoder(obs_t)

        # Select action
        if self.algo_type == "tdmpc":
            with torch.no_grad():
                action = self.planner.plan(self.z, self.device).squeeze(0).cpu().numpy()
        else:
            # SB3 models (PPO / SAC)
            action, _ = self.model.predict(self.obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        self.total_reward += float(reward)
        self.step_count += 1
        self.obs = next_obs

        # Update latent for TD-MPC2
        if self.algo_type == "tdmpc":
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                self.z = self.model.encoder(obs_t)

        # Render frame (RGB array)
        frame = self.env.render()

        metrics = {
            "label": self.label,
            "run_name": self.run_name,
            "display_name": self.display_name,
            "step": self.step_count,
            "reward": float(reward),
            "episode_reward": self.total_reward,
            "done": self.done,
            "action_magnitude": float(np.linalg.norm(action)),
        }
        
        return frame, metrics


class RolloutEngine:
    """Synchronously steps all agents and returns frames + metrics."""
    def __init__(self, model_dicts: list[dict], device: str, 
                 mppi_horizon: int = 5, mppi_samples: int = 256):
        self.agents = [
            ModelAgent(md, device, mppi_horizon, mppi_samples)
            for md in model_dicts
        ]

    def run_step(self):
        """Returns: frames (list of np.ndarray), metrics (list of dict)."""
        frames, metrics = [], []
        for agent in self.agents:
            frame, m = agent.step()
            frames.append(frame)
            metrics.append(m)
        return frames, metrics

    @property
    def labels(self):
        return [a.label for a in self.agents]

    @property
    def model_cards(self):
        return [
            {
                "label": a.label,
                "run_name": a.run_name,
                "display_name": a.display_name,
                "algorithm_name": a.algorithm_name,
                "env_title": a.env_title,
                "behavior_text": a.behavior_text,
            }
            for a in self.agents
        ]
