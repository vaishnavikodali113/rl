# Theory (Algorithms, Pseudocode, References)

This document explains the **theory behind what this project is trying to do**, while staying honest about what is implemented locally in this repo versus what is expected to come from **first-party TD‑MPC2 code**.

## 0) Terminology used in this repo

- **Observation** \(o_t\): flattened `dm_control` observation vector from `env_setup.DMCWrapper`.
- **Action** \(a_t\): continuous action vector.
- **Latent state** \(z_t\): learned embedding of \(o_t\), typically \(z_t = \mathrm{enc}(o_t)\).
- **World model / latent dynamics**: learn \(z_{t+1} = f_\theta(z_t, a_t)\) and \(r_t = g_\theta(z_t, a_t)\).
- **Planning**: choose \(a_t\) by optimizing predicted return over a short horizon using the world model (MPPI in this repo).

This repo locally implements:
- MPPI (`planning/mppi.py`)
- Simplified S4/S5/Mamba-style dynamics layers (`ssm/`)
- Optional uncertainty-aware planning truncation (`planning/info_prop.py`)

This repo *expects* TD‑MPC2 training + agent code (not included in this checkout).

## 1) TD‑MPC → TD‑MPC2 (high-level)

### 1.1 What TD‑MPC is

TD‑MPC (Temporal Difference learning for Model Predictive Control) is a model-based RL approach that:

- Learns a **task-oriented** latent world model (often called TOLD in TD‑MPC literature).
- Plans in latent space using short-horizon trajectory optimization (commonly MPPI/CEM-style).
- Uses a value function to bootstrap beyond the planning horizon.

First-party reference:
- **TD‑MPC (2022)**: Hansen, Wang, Su. “TD‑MPC: Efficient Learning Control.” (PMLR / CoRL 2022)  
  Paper: `https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf`  
  Official code: `https://github.com/nicklashansen/tdmpc`

### 1.2 What TD‑MPC2 adds

TD‑MPC2 is a follow-up designed for **scalability and robustness** across many tasks and domains.

Key high-level points (from the authors’ paper/website):
- Robust performance across many continuous-control tasks with minimal tuning.
- Scales to very large models and multi-task settings.

First-party references:
- **TD‑MPC2 paper (arXiv:2310.16828)**: “TD‑MPC2: Scalable, Robust World Models for Continuous Control”  
  Paper: `https://arxiv.org/abs/2310.16828`  
  Project page: `https://nicklashansen.github.io/td-mpc2/`  
  Official code: `https://github.com/nicklashansen/tdmpc2`

## 2) MPPI (Model Predictive Path Integral control)

MPPI is a sampling-based MPC method. At each real environment step, it:

1. Maintains a nominal action sequence \(U = (u_0, \dots, u_{H-1})\).
2. Samples \(N\) perturbed action sequences around \(U\).
3. Rolls each sequence forward through the world model to estimate returns.
4. Converts returns to weights using a temperature-scaled softmax.
5. Outputs a weighted average of the **first** action (receding horizon).
6. Shifts the nominal sequence by one step and repeats next time.

### 2.1 Pseudocode (matches `planning/mppi.py` semantics)

```text
Inputs:
  world_model with functions:
    z_next = dynamics(z, a)
    r      = reward(z, a)
  initial latent z0
  horizon H, samples N, temperature τ, discount γ
  action bounds [a_low, a_high]

State:
  nominal action sequence U_nom ∈ R^{H × act_dim}  (persistent across calls)

MPPI_PLAN(z0):
  Sample action sequences A[t, i] = clip(U_nom[t] + ε[t, i], bounds)
      for t=0..H-1, i=1..N

  For each i in 1..N:
      z ← z0
      G_i ← 0
      For t=0..H-1:
          G_i += γ^t * reward(z, A[t, i])
          z = dynamics(z, A[t, i])

  w_i = softmax(G_i / τ)  over i
  a0  = Σ_i w_i * A[0, i]     (weighted mean first action)

  U_nom[t] = Σ_i w_i * A[t, i]   for all t
  Shift U_nom left by 1; randomize last element

  return a0
```

### 2.2 Nuances in this repo’s MPPI implementation

`planning/mppi.py` includes a few practical details:

- **Persistent nominal sequence**: reduces variance and stabilizes receding-horizon behavior.
- **Dropout during planning**: it temporarily enables `torch.nn.Dropout` modules inside the model during rollouts.
  - This is used by `planning/info_prop.py` to estimate uncertainty via MC dropout disagreement.
- **Hidden state reset**: if the model’s dynamics has `reset_hidden`, MPPI resets it for the expanded batch (important for recurrent SSM dynamics).

## 3) Uncertainty-aware planning truncation (“InfoProp”)

The repo’s `planning/info_prop.py` implements an uncertainty heuristic:

- For a candidate rollout, at each step, compute uncertainty by running multiple forward passes through the dynamics with dropout enabled.
- If uncertainty exceeds a threshold:
  - Truncate the rollout and optionally bootstrap remaining return through a learned value model \(V/Q\).
  - Otherwise, continue accumulating predicted rewards and updating the latent state.

This is not a standard, universally-named algorithm; it is a project-specific mechanism inspired by the broader idea of:
- using uncertainty estimates to reduce model bias and unsafe long rollouts, and
- bootstrapping via value functions when the model is unreliable.

Important practical caveat (also stated in the code):
- This introduces a large compute overhead \(O(K \cdot H \cdot N)\) for \(K\) ensemble passes per rollout step.

## 4) SSM dynamics: S4, S5, Mamba (how they relate here)

This project’s conceptual goal is: **replace an MLP latent dynamics with an SSM-style module** to better model long-range temporal structure in the learned latent dynamics.

In this repo:
- `ssm/ssm_world_model.py:SSMDynamics` is written as a **drop-in** dynamics module with signature:
  - `z_next = dynamics(z, action)`
  - with a persistent hidden state internally.

The local S4/S5/Mamba layers here are **minimal educational implementations** to provide the correct *shape* of computation, not performance-optimized reproductions of the official papers.

### 4.1 S4 (Structured State Space Sequence model)

What S4 introduces (paper-level idea):
- A principled parameterization of linear state space models enabling efficient computation over long sequences, often via diagonal-plus-low-rank (DPLR) structure and fast kernel computations.

First-party references:
- **S4 paper (arXiv:2111.00396)**: Gu, Goel, Ré. “Efficiently Modeling Long Sequences with Structured State Spaces.”  
  Paper: `https://arxiv.org/abs/2111.00396`  
  Official code (HazyResearch): `https://github.com/state-spaces/s4`

What this repo implements:
- `ssm/s4_layer.py` implements a **diagonal stable recurrence** with discretization parameters and a `step(z_prev, u_t)` function.
- This captures “SSM-style recurrence” but does not implement full S4 convolution kernels / DPLR optimizations.

### 4.2 S5 (Simplified State Space layers)

What S5 emphasizes (paper-level idea):
- Simplified SSM layers designed to be efficient and stable, often paired with parallel scan implementations.

First-party references:
- **S5 paper (arXiv:2208.04933)**: Smith, Warrington, Linderman. “Simplified State Space Layers for Sequence Modeling.”  
  Paper: `http://arxiv.org/abs/2208.04933`  
  OpenReview: `https://openreview.net/forum?id=Ai8Hw3AXqks`  
  Official code: `https://github.com/lindermanlab/S5`

What this repo implements:
- `ssm/s5_layer.py` provides a diagonal continuous-time parameterization + discretization and a recurrence `step(...)`.
- The “parallel scan” path currently falls back to sequential computation (with a one-time warning).

### 4.3 Mamba (Selective State Space models)

What Mamba introduces (paper-level idea):
- **Selective** SSM parameters that depend on the input, enabling content-aware state updates with linear-time scaling.
- Hardware-aware kernels and architecture details for high throughput.

First-party references:
- **Mamba paper (arXiv:2312.00752)**: Gu, Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.”  
  Paper PDF: `https://arxiv.org/pdf/2312.00752`  
  Official code: `https://github.com/state-spaces/mamba`

What this repo implements:
- `ssm/mamba_layer.py` implements a minimal selective recurrence:
  - input-dependent \( \Delta t \) and \(B\) projection,
  - diagonal \(A\),
  - sequential stepping.
- It is not a full Mamba block (no mixing/gating blocks, no fused kernels).

## 5) How these pieces connect in the intended TD‑MPC2 + SSM project

The intended design pattern looks like:

1. **Encoder** maps observation \(o_t \to z_t\).
2. **Dynamics** predicts next latent given action.
   - Baseline: MLP dynamics
   - Proposed: `SSMDynamics(variant="s4"|"s5"|"mamba")`
3. **Reward model** predicts \(r_t\) from \(z_t, a_t\).
4. **Planner (MPPI)** chooses actions by rolling out the world model in latent space.
5. Optional **uncertainty truncation (InfoProp)** reduces reliance on long unreliable rollouts.

This repo already has the “planner + SSM dynamics modules” scaffolding, plus a visualization stack.
What’s missing in this checkout is the **actual TD‑MPC2 training/agent implementation** that would wire these modules into a full algorithm end-to-end.

## 6) Novelty (as a project direction)

The novelty claim (as suggested by the UI title and structure) is:

- Starting from a strong, scalable world-model RL algorithm (TD‑MPC2),
- Replace or augment its latent dynamics with **SSM-based dynamics modules** (S4/S5/Mamba-style recurrence),
- Evaluate:
  - sample efficiency and final returns (reward curves),
  - rollout prediction quality (latent MSE vs horizon),
  - planning stability (horizon ablations),
  - runtime cost (ms per rollout / update proxy benchmarks).

Even when the theory components exist separately, **making them work together** involves careful engineering:
- keeping dynamics stable under planning rollouts,
- managing recurrent hidden state when branching trajectories in MPPI,
- ensuring value bootstrapping is reliable when truncating uncertain trajectories.

## 7) Future scope (concrete, high-impact next steps)

### 7.1 Make TD‑MPC2 integration real (not just entrypoints)

- Pull in the official TD‑MPC2 implementation (`tdmpc_2` submodule) and make it importable.
- Add a local compatibility layer only if necessary (the repo currently imports `tdmpc2.compat.load_tdmpc2_agent`, but that file does not exist here).

### 7.2 Wire SSM dynamics into TD‑MPC2 training

- Replace the latent dynamics module in TD‑MPC2 with `SSMDynamics`.
- Ensure training supports:
  - `reset_hidden(batch_size, device)` for batch training
  - `snapshot_hidden` / `restore_hidden` for planning-time branching or evaluation

### 7.3 Correct evaluation signals

- Define a consistent rollout error metric:
  - latent MSE over horizons computed on held-out sequences (this repo has `evaluation/rollout_error.py`)
  - save `rollout_errors.npy` per run into `artifacts/<run>/rollout_errors.npy` so the dashboard can load it directly.

### 7.4 Make “planning stability” a first-class experiment

- Automate Phase 3 horizon ablation runs and ensure they save artifacts with the run names expected by `plot_results.py`.

### 7.5 Scale experiments

- Multi-seed evaluation, confidence intervals in plots, and standardized run metadata in `summary.json`.

## 8) References (first-party)

- TD‑MPC2 paper: `https://arxiv.org/abs/2310.16828`
- TD‑MPC2 official code: `https://github.com/nicklashansen/tdmpc2`
- TD‑MPC paper (PMLR/CoRL 2022): `https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf`
- TD‑MPC official code: `https://github.com/nicklashansen/tdmpc`
- S4 paper: `https://arxiv.org/abs/2111.00396`
- S4 official code: `https://github.com/state-spaces/s4`
- S5 paper: `http://arxiv.org/abs/2208.04933`
- S5 official code: `https://github.com/lindermanlab/S5`
- Mamba paper PDF: `https://arxiv.org/pdf/2312.00752`
- Mamba official code: `https://github.com/state-spaces/mamba`

