import time
import torch

def benchmark_update_step(model, batch_size=256, horizon=5, n_runs=100, device='cpu'):
    """Returns mean milliseconds per update step."""
    act_dim = model.reward.net[0].in_features - model.latent_dim
    
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    times = []
    
    for _ in range(n_runs):
        obs_seq = torch.randn(horizon + 1, batch_size, model.encoder.net[0].in_features, device=device)
        act_seq = torch.randn(horizon, batch_size, act_dim, device=device)
        rew_seq = torch.randn(horizon, batch_size, device=device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        z0 = model.encoder(obs_seq[0])
        latents, pred_r = model.rollout(z0, act_seq)
        loss = ((pred_r - rew_seq) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    # skip warmup
    return sum(times[10:]) / len(times[10:]) if len(times) > 10 else sum(times) / len(times)
