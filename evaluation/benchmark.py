import time
import torch

def _synchronize_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cpu":
        return
    sync_module = getattr(torch, device.type, None)
    synchronize = getattr(sync_module, "synchronize", None)
    if callable(synchronize):
        synchronize()


def benchmark_update_step(model, batch_size=256, horizon=5, n_runs=100, device='cpu'):
    """Returns mean milliseconds for the world-model rollout used in an update."""
    device = torch.device(device)
    if hasattr(model, "act"):
        obs_dim = int(model.cfg.obs_shape[model.cfg.obs][0])
        times = []
        for _ in range(n_runs):
            obs = torch.randn(obs_dim, device=device)
            _synchronize_device(device)
            start = time.perf_counter()

            with torch.no_grad():
                _ = model.act(obs, t0=False, eval_mode=True)

            _synchronize_device(device)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return sum(times[10:]) / len(times[10:]) if len(times) > 10 else sum(times) / len(times)

    act_dim = getattr(model, "action_dim", model.reward.net[0].in_features - model.latent_dim)
    obs_dim = getattr(model, "obs_dim", model.encoder.net[0].in_features)

    model.to(device)
    model.train()
    times = []
    
    for _ in range(n_runs):
        obs_seq = torch.randn(horizon + 1, batch_size, obs_dim, device=device)
        act_seq = torch.randn(horizon, batch_size, act_dim, device=device)

        _synchronize_device(device)
        start = time.perf_counter()

        with torch.no_grad():
            z0 = model.encoder(obs_seq[0])
            model.rollout(z0, act_seq)

        _synchronize_device(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times[10:]) / len(times[10:]) if len(times) > 10 else sum(times) / len(times)
