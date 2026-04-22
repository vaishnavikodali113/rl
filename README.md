# Reinforcement Learning Project

This repository now uses the upstream [`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2) codebase as a Git submodule at [`tdmpc_2/`](./tdmpc_2), while keeping the rest of the project stable through a local compatibility layer in [`tdmpc2/`](./tdmpc2).

## Current Architecture

- `tdmpc_2/`
  The upstream TD-MPC2 repository, kept as a submodule and treated as read-only.
- `tdmpc2/`
  Local compatibility wrappers that:
  - preserve the existing CLI entrypoints
  - train through the submodule without editing it
  - materialize artifacts in the repo’s historical `logs/<run>` and `artifacts/<run>` layout
  - reconstruct TD-MPC2 agents for the server and evaluation stack
- `server/`, `evaluation/`, `app.py`
  Existing project infrastructure that now consumes compatibility artifacts instead of the old custom TD-MPC2 checkpoint schema.

## Important Migration Note

This repo previously had a custom in-tree TD-MPC2 implementation with extra structured-dynamics variants (`s4`, `s5`, `mamba`).

That is no longer the active backend.

The active backend is the upstream `tdmpc_2` submodule used as-is.

Because the upstream codebase does not expose the old structured-dynamics hooks, the legacy commands and run names are preserved only for compatibility. They still invoke the canonical upstream TD-MPC2 implementation.

## Repository Layout

```text
rl/
├── app.py                      # Unified backend/dashboard launcher
├── main.py                     # Unified CLI for training, plotting, evaluation
├── env_setup.py                # dm_control -> Gymnasium adapter for SB3/server flows
├── tdmpc2/                     # Local compatibility layer around tdmpc_2
│   ├── compat.py               # Core adapter: training, artifact sync, agent loading
│   ├── train_tdmpc2.py         # Historical entrypoint, now routed to compat.py
│   └── __init__.py
├── tdmpc_2/                    # Git submodule: upstream TD-MPC2 repo
├── server/                     # FastAPI/WebSocket visualization backend
├── evaluation/                 # Offline comparison, plotting, benchmarking
├── dashboard/                  # Frontend dashboard
├── knowledge_base.md           # Human-facing technical map of the repo
└── SYSTEM_INSTRUCTIONS.md      # Starter instructions for future agents
```

## Clone And Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd rl
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install Python dependencies in your active environment:

```bash
pip install -r requirements.txt
```

## Runtime Requirements

### PPO / SAC

- Can still run on CPU, MPS, or CUDA depending on your environment.

### TD-MPC2 via `tdmpc_2`

- Requires CUDA.
- The upstream submodule hardcodes CUDA in several internal modules.
- The local compatibility layer intentionally does not edit the submodule to remove those assumptions.

If CUDA is unavailable, TD-MPC2 training/loading will fail fast with a clear error, while the rest of the repo can still run.

## Main Commands

Environment smoke test:

```bash
python main.py test
```

Train PPO:

```bash
python main.py ppo
```

Train SAC:

```bash
python main.py sac
```

Train TD-MPC2 through the submodule:

```bash
python main.py tdmpc
```

Legacy compatibility entrypoints still exist:

```bash
python main.py tdmpc-s4
python main.py tdmpc-s5
python main.py tdmpc-mamba
```

Those commands keep the old run naming convention, but the backend remains canonical upstream TD-MPC2.

Plot saved results:

```bash
python main.py plot
python main.py all-phases
```

Run the visualization backend:

```bash
python app.py serve --frontend static
```

Run backend + Vite dashboard in development:

```bash
python app.py serve --frontend dev --reload
```

## TD-MPC2 Compatibility Contract

Training through `tdmpc2.compat.train_with_vendor_backend(...)` does four things:

1. Builds an upstream-style config from the old repo CLI parameters.
2. Runs the upstream TD-MPC2 trainer from the submodule without modifying the submodule.
3. Copies/synchronizes the resulting outputs into the historical project layout.
4. Writes compatibility metadata so the rest of the codebase can keep working.

For a run named `tdmpc2_walker_mlp`, the important outputs are:

- `artifacts/tdmpc2_walker_mlp/summary.json`
- `artifacts/tdmpc2_walker_mlp/metrics.jsonl`
- `artifacts/tdmpc2_walker_mlp/model.pt`
- `logs/tdmpc2_walker_mlp/eval/evaluations.npz`

The original upstream training logs are still kept under the submodule-style work dir:

- `logs/<task>/<seed>/<exp_name>/...`

The compatibility summary points back to that original directory through `artifacts.<vendor_log_dir>`.

## Server And Evaluation Behavior

The server and evaluation stack no longer deserialize the old custom `TDMPC2Model` checkpoint format.

Instead they:

- read `summary.json`
- locate `model.pt`
- reconstruct an upstream TD-MPC2 agent through `tdmpc2.compat.load_tdmpc2_agent(...)`

For live rollouts:

- PPO/SAC still use `model.predict(...)`
- TD-MPC2 now uses the upstream agent’s `act(...)` API directly

## Files To Read First

If you are orienting in this repo, start here:

1. [`SYSTEM_INSTRUCTIONS.md`](./SYSTEM_INSTRUCTIONS.md)
2. [`knowledge_base.md`](./knowledge_base.md)
3. [`tdmpc2/compat.py`](./tdmpc2/compat.py)
4. [`main.py`](./main.py)
5. [`server/model_loader.py`](./server/model_loader.py)
6. [`server/rollout_engine.py`](./server/rollout_engine.py)

## Known Limitations

- TD-MPC2 is currently CUDA-only in this integration because the submodule is used without patching its internals.
- Legacy `s4`, `s5`, and `mamba` command names are compatibility aliases, not distinct upstream model architectures.
- The archived in-tree `tdmpc2/model.py` and `tdmpc2/trainer.py` files are no longer the active training path.

## Development Rule

Do not edit files inside `tdmpc_2/` unless you intentionally want to fork the upstream submodule.

All project-specific adaptation should happen in the main repo, primarily through:

- `tdmpc2/compat.py`
- `server/`
- `evaluation/`
- docs and metadata files
