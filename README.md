# Efficient MBRL with SSM World Models (TD-MPC2 · S4 · S5 · Mamba)

This repository contains a small RL project with:

- **Training baselines**: PPO + SAC (Stable-Baselines3).
- **Model-based RL hooks**: TD‑MPC2 entrypoints (via a git submodule; see below), and local implementations of **SSM dynamics layers** (S4/S5/Mamba-style) + **MPPI** planning utilities.
- **Visualization stack**:
  - **Backend**: FastAPI server that loads checkpoints and streams live rollouts over WebSocket.
  - **Frontend**: Vite + React dashboard that visualizes live rollouts and offline artifacts.

If you are looking for a “how it works” narrative, start with `knowledge_base.md` and `theory.md`.

## Quick start

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- **PyTorch**: `requirements.txt` pins `torch>=2.2.0` but you may want the correct CUDA wheel for your system.
- **MuJoCo + dm_control**: `dm_control` uses MuJoCo; ensure your system can render headlessly if needed.

### 2) (Optional but recommended) Initialize the TD‑MPC2 submodule

This repo expects the official TD‑MPC2 code as a submodule:

- Submodule declared in `.gitmodules` at path `tdmpc_2`
- URL: `https://github.com/nicklashansen/tdmpc2.git`

Initialize it:

```bash
git submodule update --init --recursive
```

Important:
- In the current checkout, the local folder `tdmpc2/` is present but **empty**, and imports like `from tdmpc2.train_tdmpc2 import ...` will fail until TD‑MPC2 code is available on `PYTHONPATH`.
- If you use the submodule, you’ll typically need to ensure its package is importable (e.g. add it to `PYTHONPATH`, install editable, or mirror code into `tdmpc2/`).

### 3) Run training (PPO / SAC)

```bash
python main.py ppo --env-name walker --task walk --total-steps 10000
python main.py sac --env-name cheetah --task run --total-steps 10000
```

Outputs are written to:
- `logs/<run_name>/...` (SB3 eval `.npz`, checkpoints)
- `artifacts/<run_name>/...` (metrics JSONL, `summary.json`)

The run layout is created by `run_layout.init_run_paths()` and **deletes any existing output** for the same `run_name`.

### 4) Plot offline results

```bash
python main.py plot
python main.py plot --all-phases
```

Plots are written under `artifacts/plots/...`.

### 5) Run the dashboard (backend + frontend)

Backend exposes:
- WebSocket: `/ws` (live frames + metrics)
- REST: `/health`, `/metrics/live`, `/artifacts/*`

Run both:

```bash
python app.py serve --reload
```

Frontend mode selection:
- `--frontend=auto` (default): use built static UI if `dashboard/dist` exists, otherwise start Vite dev server if `npm` is available.
- `--frontend=dev`: always start Vite dev server.
- `--frontend=static`: only serve the built UI through FastAPI.
- `--frontend=none`: backend only.

Build the frontend for static serving:

```bash
cd dashboard
npm install
npm run build
```

Then `python app.py serve` will mount `dashboard/dist` at `/` on port `8000`.

## Repository entrypoints

- **Unified CLI**: `main.py`
  - Commands: `test`, `ppo`, `sac`, `tdmpc`, `tdmpc-s4`, `tdmpc-s5`, `tdmpc-mamba`, `phase3`, `phase4`, `plot`, `all-phases`
- **Unified server launcher**: `app.py`
  - `python app.py serve` or `uvicorn app:app --reload`
- **FastAPI app**: `server/server.py`
- **Frontend**: `dashboard/src/main.tsx` → `dashboard/src/app/App.tsx`

## Where to read next

- `knowledge_base.md`: full end-to-end implementation flow (training, artifacts, server, frontend, evaluation).
- `theory.md`: algorithm theory + pseudocode + references for TD‑MPC2, MPPI, S4/S5/Mamba, and how they relate to this project.

