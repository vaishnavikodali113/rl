from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""

    def __init__(self, params, base_optimizer_class, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        kwargs.pop("rho", None)
        self.base_optimizer = base_optimizer_class(self.param_groups, **kwargs)
        # Note: self.param_groups perfectly aliases base_optimizer.param_groups here.
        # Calling super().add_param_group on SAM will break this symmetry and should be avoided.
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group.setdefault("rho", rho)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM.step requires a closure for two forward/backward passes.")
        closure = torch.enable_grad()(closure)
        loss = closure()
        
        has_grad = any(p.grad is not None for group in self.param_groups for p in group["params"])
        if not has_grad:
            raise RuntimeError("SAM.step closure did not produce any gradients. Check that loss.backward() is called.")
            
        self.first_step(zero_grad=True)
        closure()
        
        has_grad_2 = any(p.grad is not None for group in self.param_groups for p in group["params"])
        if not has_grad_2:
            raise RuntimeError("SAM.step second closure did not produce any gradients. Check that loss.backward() is called.")
            
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.stack(norms).norm(p=2)
