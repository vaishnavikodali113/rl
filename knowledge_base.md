# Reinforcement Learning Architecture Knowledge Base

This document serves as the primary technical reference for the RL codebase. It bridges the gap between state-of-the-art research (TD-MPC2, S5, Mamba) and the practical implementation of stable, long-horizon model-based reinforcement learning.

---

## 1. Theoretical Foundations

### 1.1 TD-MPC2: Latent Space World Models
TD-MPC2 (Temporal Difference Learning for Model Predictive Control) is a model-based RL algorithm that performs trajectory optimization in a **latent space**. Unlike reconstruction-heavy models (e.g., DreamerV3), TD-MPC2 is **decoder-free**: it learns representations predictive of task-relevant metrics rather than pixels.

- **Component Modules**:
    - **Encoder ($h$):** Maps observation $s_t$ to latent state $z_t$.
    - **Dynamics ($d$):** Predicts $z_{t+1}$ given $z_t$ and $a_t$.
    - **Reward ($r$):** Predicts scalar reward from $z_t, a_t$.
    - **Value ($Q$):** Estimates expected future returns (via Ensemble Q-learning).
- **Core Principle:** Training occurs via joint optimization of reward prediction, value estimation, and **latent consistency** (ensuring $d(z_t, a_t)$ matches encoded $h(s_{t+1})$).

### 1.2 Structured State-Space Models (SSMs)
The codebase replaces traditional MLPs/RNNs in the dynamics model with SSMs to solve the **vanishing gradient** problem over long horizons.

- **HiPPO (High-Order Polynomial Projection Operator):** A mathematical framework that projects the signal history onto orthogonal polynomials. It is used to initialize the $A$ matrix, allowing the model to track dependencies over thousands of steps.
- **S4/S5 (Diagonal SSMs):** 
    - **S4:** Uses a bank of independent SISO (Single-Input Single-Output) systems.
    - **S5:** Standardizes S4 into a **MIMO (Multi-Input Multi-Output)** system using a diagonalized state transition matrix. This allows for direct inter-channel interaction within a single recurrence.
- **Mamba (Selective SSM):** Introduces **Selectivity**. Parameters ($B, C, \Delta$) become input-dependent, allowing the model to "choose" what to forget or remember based on the current latent state.

### 1.3 MPPI: Model Predictive Path Integral
A derivative-free, sampling-based optimal control algorithm. 
- **Mechanism:**
    1. Sample $N$ candidate action sequences from a Gaussian distribution centered on a nominal sequence.
    2. Roll out sequences in the **latent world model** over horizon $H$.
    3. Calculate total returns (sum of predicted rewards + terminal value).
    4. Update the nominal sequence using a **Softmax-weighted average** of the samples.
- **Information Theoretic Basis:** Updates are derived from the KL-divergence between the controller distribution and the optimal (reward-weighted) distribution.

### 1.4 Stability Techniques
- **SAM (Sharpness-Aware Minimization):** Instead of just minimizing loss, SAM seeks parameters in **flat minima** where the loss is low even under local perturbations. This makes the world model more robust to out-of-distribution planning.
- **InfoProp (Uncertainty-Aware Planning):** Uses **MC-Dropout variance** to detect when the model is "hallucinating." It truncates unreliable rollouts and falls back to a learned Value function bootstrap.

---

## 2. File-by-File Technical Breakdown

### 2.1 Dynamics Layer (`ssm/`)

#### `ssm_world_model.py`
**`SSMDynamics` (Class)**
- **Role:** The high-level interface for MPPI. It manages the hidden state `_hidden` of the recurrent layers.
- **Function `forward(z, action)`:** Concatenates state and action, pushes through the selected SSM layer, and applies `out_proj`.
- **Function `reset_hidden()`:** Critical for MPPI; clears the state between rollout simulations to prevent temporal leakage.

#### `s5_layer.py`
**`S5Layer` (Class)**
- **`make_hippo_diag()`:** Implements the HiPPO-style diagonal initialization for the $A$ matrix.
- **`step(z_prev, u_t)`:** Performs the linear recurrence: $z_t = \bar{A} z_{t-1} + \bar{B} u_t$. 
- **Implementation Note:** Uses **Parallel Scan** (though sequential fallback is currently implemented) for $O(\log L)$ training complexity.

#### `mamba_layer.py`
**`MambaLayer` (Class)**
- **Mechanism:** Implements a simplified selective mechanism.
- **`step(h_prev, u_t)`:** Calculates **input-dependent $\Delta$** using a Softplus projection. This determines the "discretization step size," effectively controlling the gating of information.

---

### 2.2 Optimizer & Planning (`planning/`)

#### `mppi.py`
**`MPPI` (Class)**
- **`plan(z, device)`:** The main entry point. Orchestrates the sampling, rollout, and Softmax-weighting loop.
- **`_update_nominal_sequence()`:** Shifts the optimized plan locally so the next control step starts with a "warm" initialization.

#### `info_prop.py`
**`InfoProp` (Class)**
- **`compute_uncertainty()`:** Performs $K$ stochastic forward passes (MC-Dropout) and calculates the variance of the predictions.
- **`plan_with_truncation()`:** Injects uncertainty logic into the MPPI rollout. If variance > threshold, the transition is masked out, and the value is bootstrapped.

#### `sam_optimizer.py`
**`SAM` (Class)**
- Wraps standard optimizers (Adam). 
- **`first_step()`:** Moves parameters to the "worst-case" neighbor (maximizing local loss).
- **`second_step()`:** Computes the final gradient at that peak and moves back to origin to apply the update.

#### `sim_norm.py`
**`SimNorm` (Class)**
- Normalizes latent vectors into sub-groups (Simplex). This prevents numerical explosion in the SSM recurrence and ensures the latent space remains bounded.

---

### 2.3 System Integration (Root)

#### `main.py`
The CLI router. It defines **Phase presets**:
- `tdmpc`: Baseline MLP ($H=5$).
- `phase3`: Stability stack ($H=10$, SSM, SAM, InfoProp).

#### `env_setup.py`
**`DMCWrapper`**
- Connects `dm_control` (MuJoCo) to `gymnasium`. Handles observation flattening and spec translation.

#### `device_utils.py`
Abstraction for hardware parity. 
- **Linux/Fedora:** Prioritizes `cuda`.
- **macOS:** Prioritizes `mps`.

---

## 3. Experimental Pipeline & Log Layout

### 3.1 Training Phases
1. **Phase 0 (Baselines):** PPO and SAC benchmarks to establish the "floor."
2. **Phase 1 (MLP TD-MPC):** Short horizon ($H=5$) latent optimization.
3. **Phase 2 (SSM Dynamics):** Comparing MLP vs. S4 vs. S5 vs. Mamba at standard horizons.
4. **Phase 3 (Long-Horizon Stability):** Pushing to $H=10$ using the full stability stack (S5 + SAM + InfoProp).

### 3.2 Output Directories
- **`logs/`**: Raw Tensorboard events and checkpoint files.
- **`artifacts/`**: 
    - `metrics.jsonl`: Step-by-step history of `eval/mean_reward` and loss values.
    - `plots/`: Phase-specific performance visualizations generated by `plot_results.py`.
