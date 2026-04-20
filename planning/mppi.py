from __future__ import annotations

import torch
import torch.nn.functional as F


class MPPI:
    """Model Predictive Path Integral controller."""

    def __init__(
        self,
        world_model,
        action_dim: int,
        horizon: int = 5,
        n_samples: int = 512,
        temperature: float = 0.5,
        gamma: float = 0.99,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
        info_prop=None,
        noise_scale: float = 0.25,
    ):
        self.model = world_model
        self.action_dim = action_dim
        self.H = horizon
        self.N = n_samples
        self.temp = temperature
        self.gamma = gamma
        self.a_low = action_low
        self.a_high = action_high
        self.info_prop = info_prop
        self.noise_scale = noise_scale
        self._nominal_actions: torch.Tensor | None = None
        self._a_low_tensor: torch.Tensor | None = None
        self._a_high_tensor: torch.Tensor | None = None

    def _planning_temperature(self, total_rewards: torch.Tensor) -> torch.Tensor:
        temp = torch.as_tensor(self.temp, device=total_rewards.device, dtype=total_rewards.dtype)
        safe_temp = torch.clamp(temp, min=1e-3)
        logits = total_rewards / safe_temp
        return logits - logits.max(dim=-1, keepdim=True).values

    @torch.no_grad()
    def plan(self, z: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        batch_size, latent_dim = z.shape
        dropout_modules, dropout_states = self._enable_dropout_for_planning()
        hidden_backup = (
            self.model.dynamics.snapshot_hidden()
            if hasattr(self.model.dynamics, "snapshot_hidden")
            else None
        )
        try:
            actions = self._sample_action_sequences(batch_size, device)
            z_expand = z.unsqueeze(1).expand(-1, self.N, -1).reshape(batch_size * self.N, latent_dim)

            if hasattr(self.model.dynamics, "reset_hidden"):
                self.model.dynamics.reset_hidden(batch_size * self.N, device)

            if self.info_prop is not None:
                returns = self.info_prop.plan_with_truncation(z_expand, actions, device)
                total_rewards = returns.reshape(batch_size, self.N)
            else:
                returns = torch.zeros(batch_size * self.N, device=device)
                discounts = torch.pow(self.gamma, torch.arange(self.H, device=device, dtype=torch.float32))

                z_curr = z_expand
                for t in range(self.H):
                    rewards = self.model.reward(z_curr, actions[t])
                    returns += discounts[t] * rewards
                    z_curr = self.model.dynamics(z_curr, actions[t])
                total_rewards = returns.reshape(batch_size, self.N)
        finally:
            if hasattr(self.model.dynamics, "restore_hidden"):
                self.model.dynamics.restore_hidden(hidden_backup)
            self._restore_dropout_modes(dropout_modules, dropout_states)

        weights = F.softmax(self._planning_temperature(total_rewards), dim=-1)
        first_actions = actions[0].reshape(batch_size, self.N, self.action_dim)
        best_action = (weights.unsqueeze(-1) * first_actions).sum(dim=1)
        self._update_nominal_sequence(actions, weights, batch_size)
        return torch.clamp(best_action, min=self._to_tensor(self.a_low, device), max=self._to_tensor(self.a_high, device))

    def _enable_dropout_for_planning(self):
        modules = []
        states = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                modules.append(module)
                states.append(module.training)
                module.train(True)
        return modules, states

    @staticmethod
    def _restore_dropout_modes(modules, states) -> None:
        for module, state in zip(modules, states):
            module.train(state)

    def _sample_action_sequences(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        low = self._to_tensor(self.a_low, device).view(1, 1, self.action_dim)
        high = self._to_tensor(self.a_high, device).view(1, 1, self.action_dim)
        nominal = self._get_nominal_sequence(batch_size, device)
        noise_scale = self.noise_scale * (high - low)
        noise = torch.randn(self.H, batch_size, self.N, self.action_dim, device=device) * noise_scale.unsqueeze(2)
        candidates = nominal.unsqueeze(2) + noise
        candidates = torch.clamp(candidates, min=low.unsqueeze(2), max=high.unsqueeze(2))
        return candidates.reshape(self.H, batch_size * self.N, self.action_dim)

    def _get_nominal_sequence(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        if (
            self._nominal_actions is None
            or self._nominal_actions.shape[1] != batch_size
            or self._nominal_actions.device != device
        ):
            self._nominal_actions = torch.zeros((self.H, batch_size, self.action_dim), device=device)
        self._nominal_actions = self._nominal_actions.to(device=device)
        return self._nominal_actions

    def _update_nominal_sequence(self, actions: torch.Tensor, weights: torch.Tensor, batch_size: int) -> None:
        candidates = actions.reshape(self.H, batch_size, self.N, self.action_dim)
        weighted = (weights.unsqueeze(0).unsqueeze(-1) * candidates).sum(dim=2)
        
        if self._a_low_tensor is None or self._a_low_tensor.device != weighted.device:
            self._a_low_tensor = self._to_tensor(self.a_low, weighted.device)
        if self._a_high_tensor is None or self._a_high_tensor.device != weighted.device:
            self._a_high_tensor = self._to_tensor(self.a_high, weighted.device)
            
        weighted = torch.clamp(weighted, self._a_low_tensor, self._a_high_tensor)

        shifted = torch.empty_like(weighted)
        shifted[:-1] = weighted[1:]
        shifted[-1] = (
            torch.rand_like(weighted[-1]) * (self._a_high_tensor - self._a_low_tensor) + self._a_low_tensor
        )
        self._nominal_actions = shifted.detach()

    @staticmethod
    def _to_tensor(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=torch.float32)
        return torch.tensor(v, device=device, dtype=torch.float32)
