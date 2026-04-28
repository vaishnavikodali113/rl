# System Instructions (Repo Conventions)

This file documents the “operational contract” of the repository: directory conventions, artifact formats, and how the subsystems are expected to interact.

## Core invariants

- **Run names are canonical identifiers**:
  - `run_name` appears in `artifacts/<run_name>/summary.json`
  - `run_name` is used by plotting (`plot_results.py`) and evaluation (`evaluation/*`)
  - the dashboard backend expects specific run names via `server/config.py:ALGORITHM_REGISTRY`

- **Output layout**
  - `logs/<run_name>/` is for checkpoints and SB3 evaluation outputs
  - `artifacts/<run_name>/` is for portable visualization assets:
    - `metrics.jsonl`
    - `summary.json`
    - optionally `rollout_errors.npy`

- **Destructive behavior**
  - `run_layout.init_run_paths(run_name)` deletes `logs/<run_name>/` and `artifacts/<run_name>/` if they exist.
  - Don’t reuse a run name unless you intend to overwrite it.

## Dashboard contracts

- **WebSocket**: backend streams JSON on `/ws` with shape:
  - `labels: string[]`
  - `models: ModelCard[]`
  - `frames: base64_jpeg[]`
  - `metrics: StepMetric[]`

- **REST**:
  - `/health` must be cheap and always available (used for UX state).
  - `/artifacts/*` endpoints should be read-only and derived from files under `artifacts/`.

## TD‑MPC2 integration expectations (current repo state)

This repository references TD‑MPC2 via imports like `tdmpc2.train_tdmpc2` and `tdmpc2.compat.load_tdmpc2_agent`.

In this checkout:
- `tdmpc2/` exists but is empty.
- `.gitmodules` declares submodule `tdmpc_2` with URL `nicklashansen/tdmpc2`, but the folder is not present.

Rule:
- If you want TD‑MPC2 commands, evaluation benchmarking, or TD‑MPC2 live rollouts to work, you must fetch TD‑MPC2 code and ensure it is importable (submodule + `PYTHONPATH` / editable install / vendoring).

## Adding a new algorithm

If you add a new algorithm or model type, update in this order:

1. `server/config.py:ALGORITHM_REGISTRY` (what the live server attempts to load)
2. `server/model_loader.py` (how checkpoints are discovered and loaded)
3. `server/rollout_engine.py:ModelAgent.step()` (how actions are produced and how resets are handled)
4. `dashboard/` (optional: colors/labels mapping, UI copy)

## Artifact compatibility (recommendation)

To keep plotting and dashboard endpoints simple:

- Always write `artifacts/<run_name>/summary.json` with at least:
  - `run_name`, `algorithm`, `environment`, `total_timesteps`
  - `artifacts.metrics_jsonl`, `artifacts.summary_json`, `artifacts.model`, `artifacts.eval_npz` (when available)
- For model-based runs, write:
  - `artifacts/<run_name>/rollout_errors.npy` with shape `[horizon]`

