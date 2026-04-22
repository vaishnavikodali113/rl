# Reinforcement Learning Knowledge Base

This document describes the codebase as it exists after the TD-MPC2 submodule migration.

The key architectural fact is:

- `tdmpc_2/` is the canonical TD-MPC2 backend.
- `tdmpc2/` is now a compatibility layer owned by this repo.

## 1. System Overview

There are now three algorithm families in the repo:

1. `PPO` and `SAC`
   Local training code built around Stable-Baselines3.
2. `TD-MPC2`
   Trained and loaded through the upstream `tdmpc_2` Git submodule.
3. `Visualization / Evaluation`
   Repo-specific tooling that reads artifacts, loads policies, renders rollouts, and builds comparison outputs.

## 2. High-Level Execution Paths

### 2.1 TD-MPC2 Training

Entry path:

- `main.py`
- `tdmpc2/train_tdmpc2.py`
- `tdmpc2/compat.py::train_with_vendor_backend`

Flow:

1. The legacy CLI still calls `tdmpc2.train_tdmpc2.main(...)`.
2. That wrapper delegates to `tdmpc2.compat.train_with_vendor_backend(...)`.
3. The compatibility layer loads the upstream config from `tdmpc_2/tdmpc2/config.yaml`.
4. It imports the upstream trainer modules by adding `tdmpc_2/tdmpc2/` to `sys.path`.
5. It runs the upstream trainer without modifying submodule files.
6. It converts the resulting outputs into this repo’s historical artifact layout.

### 2.2 TD-MPC2 Loading

Entry path:

- `server/model_loader.py`
- `evaluation/main.py`
- `tdmpc2/compat.py::load_tdmpc2_agent`

Flow:

1. The loader reads a compatibility `summary.json`.
2. It rebuilds the upstream TD-MPC2 config from `summary["config"]`.
3. It instantiates the upstream TD-MPC2 agent.
4. It loads `model.pt`.
5. Server and evaluation code then use the upstream agent directly.

### 2.3 Live Rollouts

Entry path:

- `server/rollout_engine.py`

Behavior:

- PPO/SAC continue to use `model.predict(...)`.
- TD-MPC2 now prefers the upstream agent `act(...)` API.
- The old MPPI path remains only for legacy world-model objects if they ever reappear.

## 3. File-By-File Breakdown

### 3.1 Root Files

#### `main.py`

Purpose:

- Unified CLI router for training, plotting, and evaluation.

Important behavior:

- `tdmpc` and `phase3` still route through `tdmpc2.train_tdmpc2`.
- `tdmpc-s4`, `tdmpc-s5`, and `tdmpc-mamba` remain available for naming compatibility.

#### `app.py`

Purpose:

- Unified launcher for FastAPI backend and dashboard.

Important behavior:

- Detects whether to serve static frontend assets or spawn the dashboard dev server.
- Does not know TD-MPC2 internals; it only launches the app stack.

#### `env_setup.py`

Purpose:

- Local `dm_control` to Gymnasium adapter used by SB3 training and the visualization backend.

Important behavior:

- Returns flattened state observations.
- Supports vectorized and non-vectorized environments.
- This is separate from the upstream TD-MPC2 environment code inside the submodule.

#### `run_layout.py`

Purpose:

- Historical run directory manager for local training code.

Current status:

- Still relevant for PPO/SAC and older workflows.
- Not the active TD-MPC2 training layout anymore.

### 3.2 `tdmpc2/` Compatibility Layer

#### `tdmpc2/compat.py`

This is now the most important TD-MPC2 integration file in the repo.

Responsibilities:

- locate the submodule
- import upstream modules safely
- build an upstream-compatible config from repo CLI arguments
- run upstream training
- sync upstream outputs into local compatibility artifacts
- reconstruct upstream agents from stored metadata

Key functions:

- `build_vendor_cfg(...)`
  Maps old repo arguments like `env_name`, `task`, `run_name`, and `dynamics_type` into an upstream OmegaConf config.
- `train_with_vendor_backend(...)`
  Runs the upstream training stack and materializes compatibility outputs.
- `materialize_compat_artifacts(...)`
  Writes `summary.json`, `metrics.jsonl`, `model.pt`, and `evaluations.npz`.
- `load_tdmpc2_agent(...)`
  Rebuilds an upstream agent for inference.

Important constraint:

- This integration is CUDA-only because the upstream code hardcodes CUDA tensors in multiple internals.

#### `tdmpc2/train_tdmpc2.py`

Purpose:

- Legacy entrypoint retained for compatibility.

Current behavior:

- No longer trains the old in-repo world model.
- Delegates into `tdmpc2.compat.train_with_vendor_backend(...)`.

Important note:

- The old parameters for custom dynamics and planning are still accepted so old scripts do not crash.
- Those parameters are no longer forwarded into a custom TD-MPC2 implementation.

#### `tdmpc2/__init__.py`

Purpose:

- Re-exports the current public compatibility surface.

Current exports:

- `build_vendor_cfg`
- `load_tdmpc2_agent`
- `train_with_vendor_backend`

#### `tdmpc2/model.py`, `tdmpc2/trainer.py`, `tdmpc2/replay_buffer.py`

Status:

- Historical implementation files from the older custom TD-MPC2 stack.

Current role:

- They are no longer on the active training or loading path after the submodule migration.
- Treat them as archived reference code unless you intentionally plan a larger cleanup.

### 3.3 Submodule: `tdmpc_2/`

#### `tdmpc_2/tdmpc2/train.py`

Purpose:

- Upstream Hydra training entrypoint.

Why this repo does not call it directly:

- It assumes its own source tree is the import root.
- It asserts CUDA.
- It writes logs in the upstream directory structure, not this repo’s historical artifact format.

The compatibility layer instead imports the same upstream components and orchestrates them from the parent repo.

#### `tdmpc_2/tdmpc2/tdmpc2.py`

Purpose:

- Upstream TD-MPC2 agent implementation.

Important behavior:

- Provides `act(...)`, `update(...)`, `save(...)`, and `load(...)`.
- Hardcodes CUDA device usage internally.
- Is the runtime object used by server/evaluation after reconstruction.

#### `tdmpc_2/tdmpc2/common/`

Important modules:

- `parser.py`
  Converts and enriches Hydra configs.
- `buffer.py`
  Upstream replay buffer with CUDA-aware storage policy.
- `logger.py`
  Writes upstream logs and optional `eval.csv`.
- `layers.py`, `world_model.py`, `scale.py`
  Core model building blocks.

#### `tdmpc_2/tdmpc2/envs/`

Purpose:

- Upstream task factory and wrappers for TD-MPC2 training/evaluation.

Important distinction:

- This environment stack is used during TD-MPC2 training.
- The repo’s own `env_setup.py` is used for SB3 flows and visualization.

### 3.4 `server/`

#### `server/config.py`

Purpose:

- Registry of displayable algorithms and artifact folder names.

Important behavior:

- Keeps the old run labels, including legacy TD-MPC2 variant names.
- Those labels still matter because the server UI expects stable artifact directories.

#### `server/model_loader.py`

Purpose:

- Loads PPO, SAC, and TD-MPC2 models for the visualization backend.

Current TD-MPC2 behavior:

- Finds `model.pt`
- Finds the matching `summary.json`
- Reconstructs an upstream TD-MPC2 agent through `tdmpc2.compat.load_tdmpc2_agent(...)`

#### `server/rollout_engine.py`

Purpose:

- Uniform runtime wrapper around all loaded policies.

Important behavior:

- For PPO/SAC: calls `predict(...)`.
- For TD-MPC2 agent objects: calls `act(...)`.
- Falls back to the old MPPI planner only if a legacy world-model object is passed in.

### 3.5 `evaluation/`

#### `evaluation/main.py`

Purpose:

- Offline evaluation entrypoint for saved runs.

Current behavior:

- Discovers runs from `artifacts/*/summary.json`.
- Reconstructs TD-MPC2 agents from compatibility metadata.
- Produces comparison CSVs and a short markdown report.

#### `evaluation/compare_plots.py`

Purpose:

- Reads `summary.json`, `metrics.jsonl`, and `evaluations.npz`.

Important behavior:

- Does not know about the submodule directly.
- It only depends on the compatibility artifact contract.

#### `evaluation/eval_runner.py`

Purpose:

- Shared evaluation loop.

Current behavior:

- Uses `model.act(...)` when the loaded object is an upstream TD-MPC2 agent.
- Uses MPPI/world-model planning only for legacy objects.

#### `evaluation/benchmark.py`

Purpose:

- Lightweight performance benchmarking helper.

Current behavior:

- Benchmarks `act(...)` latency for upstream TD-MPC2 agents.
- Benchmarks rollout latency for legacy world-model objects.

### 3.6 `dashboard/`

Purpose:

- Frontend only.

Current relationship to the migration:

- It continues to read whatever the backend exposes.
- The migration impact is indirect through `server/`.

## 4. Artifact Contract

For a compatibility run `artifacts/<run_name>/`, these files are expected:

- `summary.json`
  Canonical metadata record for loaders and evaluators.
- `metrics.jsonl`
  Line-delimited metrics generated from upstream `eval.csv`.
- `model.pt`
  Upstream TD-MPC2 agent checkpoint copied into the historical artifact path.

For `logs/<run_name>/`, these files are expected:

- `eval/evaluations.npz`
  Compatibility evaluation curve file.
- `best/best_model.pt`
  Convenience copy for older loader assumptions.
- `models/final.pt`
  Convenience copy of the final upstream checkpoint.

## 5. Architectural Truths To Preserve

1. Do not edit `tdmpc_2/` for project-specific integration work.
2. Keep the compatibility boundary in `tdmpc2/compat.py`.
3. Preserve `summary.json` as the source of truth for TD-MPC2 reconstruction.
4. Preserve historical run names unless there is a deliberate migration plan.
5. Be explicit that legacy `s4`/`s5`/`mamba` labels are compatibility aliases, not upstream feature flags.

## 6. Risks And Gotchas

- CUDA dependency:
  TD-MPC2 will fail without CUDA because the submodule is used unmodified.
- Dual environment stacks:
  `env_setup.py` and `tdmpc_2/tdmpc2/envs/` serve different purposes.
- Legacy files:
  The old in-tree TD-MPC2 implementation still exists, which can mislead new contributors if they do not start from `tdmpc2/compat.py`.
- Artifact assumptions:
  Server and evaluation code now depend on the compatibility summary schema rather than the old custom checkpoint schema.
