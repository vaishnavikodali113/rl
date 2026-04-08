import time
import torch

def benchmark_update_step(model, obs_dim, action_dim, batch_size=256, horizon=5, n_runs=100, device='cpu'):
    """Returns mean milliseconds per update step."""
    obs_seq = torch.randn(horizon + 1, batch_size, obs_dim).to(device)
    act_seq = torch.randn(horizon, batch_size, action_dim).to(device)
    rew_seq = torch.randn(horizon, batch_size).to(device)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    times = []
    
    # Warmup
    for _ in range(10):
        z0 = model.encoder(obs_seq[0])
        latents, pred_r = model.rollout(z0, act_seq)
        loss = ((pred_r - rew_seq) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(n_runs):
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
        
    return sum(times) / len(times)
