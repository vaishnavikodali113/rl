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
    - **Value ($Q$):** Estimates expected future returns via an ensemble of Q-functions ($Q_1, Q_2$).
- **Core Principle:** Training occurs via joint optimization of reward prediction, value estimation, and **latent consistency** (ensuring $d(z_t, a_t)$ matches the target encoded state $h(s_{t+1})$), typically using discrete regression over log-transformed targets.
- **Latent Interface Contracts**:
    - **Dimensionality:** The latent state $z_t$ has a fixed dimension defined by `latent_dim`.
    - **Normalisation (SimNorm):** Latents are normalized using **Simplicial Normalization**, projecting feature blocks onto a probability simplex (group-wise Softmax).
    - **Stochasticity:** While the dynamics are functionally deterministic, **MC-Dropout** (p=0.1) is used during training and planning to provide uncertainty estimates.

### 1.2 Structured State-Space Models (SSMs)
The codebase replaces traditional MLPs/RNNs in the dynamics model with SSMs to solve the **vanishing gradient** problem over long horizons.

- **HiPPO (High-Order Polynomial Projection Operator):** A mathematical framework that projects the signal history onto orthogonal polynomials. It is used to initialize the $A$ matrix, allowing the model to track dependencies over thousands of steps.
- **S4/S5 (Diagonal SSMs):** 
    - **S4:** Uses a bank of independent SISO (Single-Input Single-Output) systems.
    - **S5:** Standardizes the state transition matrix $A$ into a **diagonal MIMO** system. It uses a **Zero-Order Hold (ZOH)** discretization to compute $\bar{A} = \exp(\Delta A)$.
- **Mamba (Selective SSM):** Introduces **Selectivity**. Discretization ($\Delta$) is computed via a Softplus projection of the input $u_t$, allowing the model to modulate information flow (gating) based on the current latent state.

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
- **Role:** High-level interface for MPPI. It manages the recurrent hidden state `_hidden`.
- **Function `forward(z, action)`:** 
    1. Concatenates $z$ and $a$.
    2. Performs recurrence via `ssm.step(hidden, u)`.
    3. Prevents graph growth via `.detach()` during planning (non-grad).
    4. Applies **Dropout** (p=0.1) and **SimNorm**.
- **Hidden State Management:** 
    - `reset_hidden(batch_size)`: Initializes `_hidden` to zeros. MUST be called at the start of each MPPI rollout batch and each training sequence.
    - `detach_hidden()`: Used during training to facilitate Truncated Backpropagation Through Time (T-BPTT).

#### `s5_layer.py`
**`S5Layer` (Class)**
- **Mechanism:** Linear transition $h_t = \bar{A} h_{t-1} + \bar{B} u_t$.
- **Initialization:** `make_hippo_diag()` uses a Hippo-style initialization for the log-negative diagonal of $A$.
- **Discretization:** Uses Zero-Order Hold (ZOH). $\bar{A}$ and $\bar{B}$ are pre-computed analytically before the step.

#### `mamba_layer.py`
**`MambaLayer` (Class)**
- **Mechanism:** $h_t = \exp(\Delta A) h_{t-1} + (\Delta B) u_t$.
- **Selectivity:** $\Delta$ (discretization step) is predicted per-step using `dt_proj` (Softplus). $B$ is also input-dependent (`b_proj`), allowing selective information intake based on $z_t, a_t$.

---

### 2.2 Optimizer & Planning (`planning/`)

#### `mppi.py`
**`MPPI` (Class)**
- **`plan(z, device)`:** 
    1. Samples $N=512$ sequences from a nominal distribution.
    2. Resets dynamics hidden state for $B \times N$ samples.
    3. Scores trajectories using discounted predicted rewards.
- **`_update_nominal_sequence()`:** Updates the mean of the sampling distribution using a **Softmax-weighted average** of elites. The sequence is then **time-shifted** (warm-start) for the next step.

#### `info_prop.py`
**`InfoProp` (Class)**
- **`compute_uncertainty()`:** Measure variance across $K=5$ stochastic forward passes (via MC-Dropout). Uncertainty is defined as the mean variance of the predicted latent transition $z_{t+1}$.
- **`plan_with_truncation()`:** 
    - Truncates rollouts where `uncertainty / running_var > threshold`.
    - **Bootstrap:** Replaces the remainder of truncated trajectories with the average of the Ensemble Q-functions ($Q_1, Q_2$).
    - **Avoidance Bias:** If Value functions are not yet "ready" (unreliable), the reward is set to 0, biasing the planner away from unknown regions.

#### `sam_optimizer.py`
**`SAM` (Class)**
- Wraps standard optimizers (Adam). 
- **`first_step()`:** Moves parameters to the "worst-case" neighbor (maximizing local loss).
- **`second_step()`:** Computes the final gradient at that peak and moves back to origin to apply the update.

#### `sim_norm.py`
**`SimNorm` (Class)**
- **Geometry:** Projects the latent vector into **simplex blocks**. It reshapes the input into groups of size `simnorm_dim=8` and applies `softmax` across the last dimension. This ensures the latent space remains bounded and promotes sparse feature activation.

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

---

## 4. Conceptual Alignment & Technical Risks

### 4.1 Canonical TD-MPC2 vs. This Implementation
| Feature | Canonical TD-MPC2 | This Project |
| :--- | :--- | :--- |
| **Planner** | Gradient-based (CEM/MPPI) | Pure MPPI (Sampling-based) |
| **Dynamics** | MLP/RNN | Structured State-Space (S5/Mamba) |
| **Latent Targets** | One-step consistency | Recurrent sequence consistency |
| **Robustness** | LayerNorm/Ensemble | SAM + InfoProp (Truncation) |

### 4.2 Engineering Risks & Mitigations
- **Hidden State Contamination:** MPPI requires $N$ independent latent rollouts. Failure to call `reset_hidden` with the correct batch size before `plan()` leads to state leakage across samples.
- **Train-Plan Mismatch:** MC-Dropout must be explicitly toggled in `InfoProp` while the rest of the model remains in `.eval()` mode to ensure uncertainty is calibrated.
- **Bootstrap Accuracy:** Truncation relies on a calibrated Q-ensemble. If the Value heads are trained on teacher-forced latents but queried on drifted SSM rollouts, the bootstrap target may be overconfident.
- **Numerical Stability:** SSM recurrent chains can explode. `SimNorm` is the primary safeguard; if the latent dimension is not divisible by the group size (8), initialization will fail.

---

## 5. Experimental Pipeline & Log Layout

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
