# System Instructions For New Agents

Start from these rules when working in this repository.

## 1. Ground Truth

- The active TD-MPC2 backend is the Git submodule at `tdmpc_2/`.
- The local `tdmpc2/` package is a compatibility layer, not the canonical algorithm implementation.
- Do not assume the old in-tree `tdmpc2/model.py` and `tdmpc2/trainer.py` are still active.

## 2. Do Not Edit The Submodule For Repo Integration

Unless the task explicitly asks for an upstream fork, treat `tdmpc_2/` as read-only.

All repo-specific adaptation belongs in:

- `tdmpc2/compat.py`
- `server/`
- `evaluation/`
- docs and metadata files

## 3. Files To Read First

Read these before making TD-MPC2 changes:

1. `README.md`
2. `knowledge_base.md`
3. `tdmpc2/compat.py`
4. `server/model_loader.py`
5. `server/rollout_engine.py`
6. `evaluation/main.py`

## 4. TD-MPC2 Runtime Rules

- The submodule integration is CUDA-only.
- If CUDA is unavailable, TD-MPC2 training/loading should fail clearly rather than silently degrade.
- PPO and SAC are still separate local flows and may work on CPU/MPS.

## 5. Artifact Contract

When TD-MPC2 training runs through the compatibility layer, preserve these outputs:

- `artifacts/<run_name>/summary.json`
- `artifacts/<run_name>/metrics.jsonl`
- `artifacts/<run_name>/model.pt`
- `logs/<run_name>/eval/evaluations.npz`

`summary.json` is the source of truth for reconstructing TD-MPC2 agents later.

## 6. Legacy Naming

- `tdmpc`, `tdmpc-s4`, `tdmpc-s5`, and `tdmpc-mamba` commands still exist.
- The upstream submodule does not implement the old custom structured-dynamics variants from this repo.
- Keep those legacy names only for compatibility unless a deliberate migration changes them.

If you touch docs or UX around these commands, be explicit that the non-`mlp` labels are compatibility aliases.

## 7. Preferred Debug Path

For TD-MPC2 issues, check in this order:

1. `tdmpc2/compat.py`
2. compatibility artifacts in `artifacts/<run_name>/summary.json`
3. server/evaluation loaders
4. only then inspect the submodule internals in `tdmpc_2/`

## 8. Safe Assumptions

- Existing dashboards and evaluation code should keep working through compatibility artifacts.
- New work should preserve stable run names unless there is a migration plan.
- If you need to change the TD-MPC2 integration boundary, document it in both `README.md` and `knowledge_base.md`.
