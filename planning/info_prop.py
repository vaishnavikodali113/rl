from __future__ import annotations

import torch


class InfoProp:
    """Uncertainty-aware rollout truncation using MC dropout disagreement.

    When uncertainty exceeds the threshold, the trajectory is *truncated* and
    bootstrapped through the value function for the remainder. This assumes the
    value model is calibrated on the latent space used here.

    WARNING: MC Dropout computes `K` passes per rollout step for all `N` samples.
    This creates an O(K * H * N) computational overhead per MPPI planning step, 
    so it expects high-capacity GPU processing. Furthermore, trajectories that 
    fail reliability checks during bootstrap fallback default to a flat reward of 0 
    for their remainder; this biases the planner heavily to *avoid* uncertain regions.
    """

    def __init__(
        self,
        world_model,
        n_ensemble: int = 5,
        uncertainty_threshold: float = 0.1,
        gamma: float = 0.99,
    ):
        self.model = world_model
        self.K = n_ensemble
        self.threshold = uncertainty_threshold
        self.gamma = gamma
        self.running_var = None

    @torch.no_grad()
    def compute_uncertainty(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dynamics = self.model.dynamics
        if hasattr(dynamics, "_hidden"):
            hidden_backup = dynamics._hidden.clone() if dynamics._hidden is not None else None
        else:
            hidden_backup = None

        dropout_modules = []
        dropout_states = []
        for module in dynamics.modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_modules.append(module)
                dropout_states.append(module.training)
                module.train(True)

        preds = []
        try:
            for _ in range(self.K):
                if hidden_backup is not None and hasattr(dynamics, "_hidden"):
                    dynamics._hidden = hidden_backup.clone()
                preds.append(dynamics(z, a))
        finally:
            for module, state in zip(dropout_modules, dropout_states):
                module.train(state)
            if hasattr(dynamics, "_hidden"):
                dynamics._hidden = hidden_backup.clone() if hidden_backup is not None else None

        stacked = torch.stack(preds, dim=0)
        var_est = stacked.var(dim=0).mean(dim=-1)
        if self.running_var is None:
            self.running_var = var_est.mean().item()
        else:
            self.running_var = 0.99 * self.running_var + 0.01 * var_est.mean().item()

        return var_est / (self.running_var + 1e-8)

    @torch.no_grad()
    def plan_with_truncation(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        batch_size = actions.shape[1]
        horizon = actions.shape[0]
        total = torch.zeros(batch_size, device=device)
        z = z0
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        discounts = torch.pow(self.gamma, torch.arange(horizon, device=device, dtype=torch.float32))

        if hasattr(self.model.dynamics, "reset_hidden"):
            self.model.dynamics.reset_hidden(batch_size=batch_size, device=device)

        for t in range(horizon):
            if not active.any():
                break

            a_t = actions[t]
            uncertainty = self.compute_uncertainty(z, a_t)
            uncertain = (uncertainty > self.threshold) & active

            if uncertain.any():
                bootstrap_check = getattr(self.model, "is_value_bootstrap_reliable", None)
                if callable(bootstrap_check):
                    reliable = bool(bootstrap_check(z[uncertain]))
                else:
                    reliable = bool(getattr(self.model, "value_bootstrap_ready", False))
                if reliable:
                    q1, q2 = self.model.value(z[uncertain], a_t[uncertain])
                    total[uncertain] += discounts[t] * 0.5 * (q1 + q2)
                # If not reliable, trajectory defaults to zero reward for the uncertain remainder.
                # NOTE: This creates a hard avoidance bias against unexplored/uncertain regions.

            certain = active & (~uncertain)
            if certain.any():
                rewards = self.model.reward(z[certain], a_t[certain])
                total[certain] += discounts[t] * rewards
                
                # Performance Optimization: Mask the dynamics call to skip dead trajectories.
                # We save the full hidden state, slice it for the active batch,
                # compute the step, and then scatter the results back.
                full_hidden = None
                if hasattr(self.model.dynamics, "_hidden") and self.model.dynamics._hidden is not None:
                    full_hidden = self.model.dynamics._hidden
                    self.model.dynamics._hidden = full_hidden[certain]
                
                # Compute next state only for certain trajectories
                z_next_subset = self.model.dynamics(z[certain], a_t[certain])
                
                # Update latent state
                z = z.clone() # Ensure no side-effects on the original z tensor
                z[certain] = z_next_subset
                
                # Restore and update hidden state
                if full_hidden is not None:
                    full_hidden[certain] = self.model.dynamics._hidden
                    self.model.dynamics._hidden = full_hidden

            active = certain

        return total
