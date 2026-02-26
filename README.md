# Efficient Model-Based Reinforcement Learning with Structured State-Space World Models

> B.Tech AIML — Reinforcement Learning Course Project  
> Platform: macOS (Apple Silicon M4) · Python 3.13 · MuJoCo 3.5 · Stable-Baselines3

---

## Project Overview

This project evaluates whether structured state-space (SSM-inspired) dynamics models can improve planning stability and sample efficiency in **TD-MPC2**, compared to standard MLP-based dynamics, under consumer-grade hardware constraints.

The focus is on **state-based continuous control tasks** in MuJoCo via DeepMind Control Suite (dm_control), with PPO and SAC as model-free baselines.

---

## Environments

| Environment | Task | Obs Shape | Action Shape |
|-------------|------|-----------|--------------|
| Walker | walk | (24,) | (6,) |
| Cheetah | run | (17,) | (6,) |
| Hopper | hop | (15,) | (4,) |

All environments use **state observations only** (no pixels).

---

## Project Structure

```
rl/
├── env_setup.py          # dm_control → Gymnasium wrapper (no dmc2gym)
├── train_ppo_mac.py      # PPO baseline (walker_walk, 300k steps)
├── train_sac_mac.py      # SAC baseline (cheetah_run, 500k steps)
├── plot_results.py       # Matplotlib learning curve plots
├── requirements.txt      # All dependencies
├── logs/
│   ├── ppo_walker_mac/   # PPO checkpoints + eval logs
│   └── sac_cheetah_mac/  # SAC checkpoints + eval logs
└── README.md
```

---

## Setup (Apple Silicon — M4 Mac)

### Prerequisites

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python glfw git
```

### Create Virtual Environment

```bash
python3 -m venv rl-env
source rl-env/bin/activate
pip install --upgrade pip setuptools wheel
```

### Install Dependencies

```bash
# Core RL and physics
pip install torch torchvision torchaudio
pip install "stable-baselines3[extra]" gymnasium
pip install numpy matplotlib tensorboard seaborn pandas
pip install mujoco
pip install --no-deps dm_control
pip install dm-env dm-tree glfw lxml scipy requests pyopengl
pip install gym==0.26.2
pip install git+https://github.com/denisyarats/dmc2gym.git --no-deps
pip install "shimmy[dm-control]"
```

> **Note:** `labmaze` is intentionally skipped — it requires Bazel and is not needed for Walker, Cheetah, or Hopper tasks.

### Verify Installation

```bash
python3 - <<'PY'
import torch, mujoco, stable_baselines3
from env_setup import make_env
print("Torch:", torch.__version__, "| MPS:", torch.backends.mps.is_available())
print("MuJoCo:", mujoco.__version__)
env = make_env("walker", "walk")
obs = env.reset()
print("env OK — obs shape:", obs.shape)
PY
```

Expected output:
```
Torch: 2.10.0 | MPS: True
MuJoCo: 3.5.0
env OK — obs shape: (1, 24)
```

---

## Environment Wrapper

Since `dmc2gym` is incompatible with NumPy 2.0 and Python 3.13, a custom wrapper is used that talks directly to `dm_control`:

```python
# env_setup.py
class DMCWrapper(gym.Env):
    # Wraps dm_control suite.load() into a Gymnasium-compatible API
    # Supports reset(), step(), observation_space, action_space
```

---

## Training

### PPO Baseline (Walker — Walk)

```bash
python3 train_ppo_mac.py
```

| Setting | Value |
|---------|-------|
| Algorithm | PPO |
| Environment | walker_walk |
| Total timesteps | 300,000 |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| Device | MPS (Apple Silicon) |

### SAC Baseline (Cheetah — Run)

```bash
python3 train_sac_mac.py
```

| Setting | Value |
|---------|-------|
| Algorithm | SAC |
| Environment | cheetah_run |
| Total timesteps | 500,000 |
| Learning rate | 3e-4 |
| buffer_size | 500,000 |
| batch_size | 256 |
| Device | MPS (Apple Silicon) |

---

## Results (Baselines)

### PPO — Walker Walk

| Timesteps | Mean Reward |
|-----------|-------------|
| 10,000 | 23.2 |
| 20,000 | 32.5 |
| 30,000 | 39.9 |
| 40,000 | 60.3 |
| 290,000 | 184.3 (best) |
| 300,000 | 174.3 |

### SAC — Cheetah Run

| Timesteps | Mean Reward |
|-----------|-------------|
| 20,000 | 16.7 |
| 40,000 | 44.7 |
| 60,000 | 73.3 |
| 72,000 | climbing ↑ |

---

## Plot Results

After both training runs complete:

```bash
python3 plot_results.py
```

Saves learning curves to `./logs/baseline_results.png`.

---

## Hardware

| Component | Spec |
|-----------|------|
| Device | MacBook Air M4 |
| Backend | Apple Metal (MPS) |
| RAM | 16 GB |
| Python | 3.13 |
| MuJoCo | 3.5.0 |
| PyTorch | 2.10.0 |

---

## Work Division

| Student | Responsibility |
|---------|---------------|
| A | TD-MPC2 study, baseline implementation, debugging |
| B | MuJoCo / dm_control setup, PPO & SAC baselines ✅ |
| C | SSM dynamics implementation, parameter matching |
| D | Experiment execution, plotting, analysis, report |

---

## Known Issues & Fixes

| Issue | Fix Applied |
|-------|-------------|
| `labmaze` fails to build (requires Bazel) | Installed `dm_control` with `--no-deps`, skipped `labmaze` |
| `dmc2gym` incompatible with NumPy 2.0 | Replaced with custom `DMCWrapper` using `dm_control` directly |
| `gym.envs.registry.env_specs` removed | Patched `dmc2gym/__init__.py` |
| TensorBoard broken on Python 3.13 + NumPy 2.0 | Using `plot_results.py` with matplotlib instead |
| `gym-dmcontrol` not available for Python 3.13 ARM | Replaced with `dmc2gym` from GitHub |

---

## Next Steps

- [ ] TD-MPC2 baseline implementation (Student A)
- [ ] SSM dynamics integration (Student C)
- [ ] Ablation: planning horizon 5 vs 10
- [ ] Multi-seed experiments (3 seeds)
- [ ] Final comparison plots and report

---

## References

- [MuJoCo](https://mujoco.org/)
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [TD-MPC2](https://www.tdmpc2.com/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
