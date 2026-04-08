import time
import torch

def benchmark_update_step(model, batch_size=256, horizon=5, n_runs=100, device='cpu'):
    """Returns mean milliseconds per update step."""
    obs_seq = torch.randn(horizon + 1, batch_size, model.encoder.net[0].in_features).to(device)
    # The action dimension can be inferred. The reward head takes [latent_dim + action_dim]
    act_dim = model.reward.net[0].in_features - model.latent_dim
    act_seq = torch.randn(horizon, batch_size, act_dim).to(device)
    rew_seq = torch.randn(horizon, batch_size).to(device)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        z0 = model.encoder(obs_seq[0])
        latents, pred_r = model.rollout(z0, act_seq)
        loss = ((pred_r - rew_seq) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    # skip warmup
    return sum(times[10:]) / len(times[10:]) if len(times) > 10 else sum(times) / len(times)
