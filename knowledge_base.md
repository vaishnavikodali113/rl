# Knowledge Base (Implementation-First)

This document is a **non-redundant, direct-language** walkthrough of the codebase: what each subsystem does, how data flows through it, and the non-obvious nuances you need to run, modify, or extend it.

## 1) What this project is (as implemented)

The repo is an RL experimentation + visualization stack:

- **Baselines**: PPO and SAC training using Stable-Baselines3.
- **Model-based lane**: entrypoints for TD‑MPC2 experiments, plus local code for:
  - **MPPI** planning (`planning/mppi.py`)
  - **SSM dynamics modules** (S4/S5/Mamba-inspired) (`ssm/`)
  - **InfoProp** uncertainty-aware truncation for planning (`planning/info_prop.py`)
  - **SAM** optimizer wrapper (`planning/sam_optimizer.py`)
- **Artifacts pipeline**: writes JSONL metrics + `summary.json` under `artifacts/<run_name>/` and checkpoints under `logs/<run_name>/`.
- **Visualization stack**:
  - **Backend** (`server/`): loads checkpoints, runs live rollouts, streams frames+metrics over WebSocket, serves artifact data via REST.
  - **Frontend** (`dashboard/`): a Vite/React dashboard consuming those endpoints.

### Critical repo-state nuance (TD‑MPC2 code is not present)

In this checkout:

- Folder `tdmpc2/` exists but is **empty**.
- Multiple modules import `tdmpc2.*` (e.g. `main.py`, `train_tdmpc2_s5.py`, `server/model_loader.py`, `evaluation/main.py`).
- `.gitmodules` declares a submodule `tdmpc_2` pointing to the official repo `nicklashansen/tdmpc2`, but the submodule folder is **not present** here.

Practical implication:
- **PPO/SAC training, plotting, and the dashboard can work** (assuming dependencies + artifacts exist).
- **TD‑MPC2 commands will fail** unless you bring in TD‑MPC2 code and ensure imports resolve.

## 2) Top-level entrypoints and “what runs what”

### 2.1 `main.py` (unified CLI)

`main.py` is the CLI dispatcher using `argparse`.

Commands:
- **`test`** → `test_env.main()` (smoke-test env wrapper).
- **`ppo`** → `train_ppo_mac.main(...)` (walker & cheetah defaults if not overridden).
- **`sac`** → `train_sac_mac.main(...)`.
- **`plot` / `all-phases`** → `plot_results.main(...)`.
- **`phase4`** → `evaluation.main.main()` (offline evaluation report/plots from artifacts).
- **TD‑MPC2-related**: `tdmpc`, `tdmpc-s4`, `tdmpc-s5`, `tdmpc-mamba`, `phase3`
  - These import `tdmpc2.train_tdmpc2` or wrappers like `train_tdmpc2_s5.py`.
  - They require TD‑MPC2 code to be available.

Non-obvious behavior:
- For `ppo` and `sac`, if `--env-name` is not provided, it loops over `["cheetah", "walker"]`.

### 2.2 `app.py` (unified backend + frontend launcher)

`app.py` supports:

- `python app.py serve`: start FastAPI backend, and optionally a frontend dev server.
- `uvicorn app:app --reload`: works because `app` is exported as a lazy ASGI wrapper (`LazyASGIApp`) to avoid importing backend dependencies until needed.

Frontend modes (`--frontend`):
- **`auto`**: serve built static UI if `dashboard/dist` exists; else start dev server if `npm` exists; else none.
- **`dev`**: always start `npm run dev` in `dashboard/`.
- **`static`**: backend only; serves `dashboard/dist` if it exists.
- **`none`**: backend only.

In dev mode, `app.py` sets:
- `VITE_API_BASE_URL=http://<public_host>:<backend_port>`
- `VITE_WS_BASE_URL=ws://<public_host>:<backend_port>`

These are consumed by dashboard `lib/api` (not shown here, but implied by `use-websocket.ts`/`use-artifact.ts`).

### 2.3 `server/server.py` (FastAPI app)

Backend responsibilities:
- **Model loading** at startup (lifespan context manager).
- **Live rollout loop** that steps all loaded models at `STREAM_FPS` and broadcasts payloads to WebSocket clients.
- **REST endpoints** that read artifact files on disk for charts/tables.
- **Optional static UI serving** if `dashboard/dist` exists.

Key globals:
- `engine: RolloutEngine | None`
- `tracker: MetricsTracker | None`
- `clients: set[WebSocket]`
- `latest_payload: dict | None`
- `startup_error`, `stream_error`: strings shown in `/health` and surfaced in UI.

WebSocket `/ws`:
- Server accepts and adds the socket to `clients`.
- If `latest_payload` exists, it is sent immediately.
- Then it loops on `ws.receive_text()` to keep the connection alive (frontend does not need to send meaningful data).

Live rollout loop (`rollout_loop()`):
- When no clients are connected, it sleeps and does not step envs (reduces compute).
- On each tick:
  - `frames, metrics = engine.run_step()`
  - `tracker.update(metrics)` (keeps a rolling window)
  - frames are JPEG-encoded to base64 (`server/video_stream.py`)
  - payload shape:
    - `labels`: agent labels (string list)
    - `models`: “model cards” metadata list
    - `frames`: base64 JPEG list (aligned with `models`)
    - `metrics`: list of per-model metrics dicts

REST endpoints:
- `GET /metrics/live` → rolling window snapshot from `MetricsTracker`.
- `GET /artifacts/reward-curves` → parses `artifacts/*/metrics.jsonl`.
- `GET /artifacts/rollout-errors` → loads `artifacts/*/rollout_errors.npy` OR falls back to `artifacts/evaluation_pngs/rollout_error_stats.csv`.
- `GET /artifacts/comparison-table` → reads `artifacts/evaluation_pngs/comparison_table.csv`.
- `GET /health` → `{status, models, startup_error, stream_error}`.

Static UI:
- If `dashboard/dist` exists, mounted at `/` via `StaticFiles(..., html=True)`.

## 3) Training pipeline (PPO / SAC)

### 3.1 Environment wrapper (`env_setup.py`)

- Uses `dm_control.suite.load(domain_name, task_name, task_kwargs={"random": seed})`.
- Flattens `dm_control` dict observations into one `np.float32` vector.
- Exposes a Gymnasium-style API via `DMCWrapper`.
- `make_env(..., vectorized=True)` returns `DummyVecEnv([_init])` for SB3 compatibility.
- For live rendering, `vectorized=False, render_mode="rgb_array"` is used (see backend rollouts).

### 3.2 Output layout (`run_layout.py`)

`init_run_paths(run_name)` creates the canonical directory structure:

- `logs/<run_name>/`
  - `best/` (EvalCallback best model checkpoints)
  - `eval/` (SB3 `evaluations.npz`)
  - `final_model` (saved via `model.save`)
- `artifacts/<run_name>/`
  - `metrics.jsonl` (periodic logger dumps)
  - `summary.json` (run metadata)

Important nuance:
- If `logs/<run_name>` or `artifacts/<run_name>` exist, they are **deleted** (`shutil.rmtree`) before creating new outputs.

### 3.3 Metrics logging (`artifact_logging.py`)

- `JsonLinesMetricCallback` writes one JSON object per logging event:
  - `timestamp_utc`
  - `timesteps`
  - `metrics`: snapshot of `self.logger.name_to_value` (SB3 logger keys)
- Used by both PPO and SAC training scripts.

### 3.4 PPO training (`train_ppo_mac.py`)

Defaults:
- `walker`: task `walk`, run name `ppo_walker`, steps `10_000`
- `cheetah`: task `run`, run name `ppo_cheetah`, steps `10_000`

Model config:
- Policy: `"MlpPolicy"`
- Key PPO params: `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`

Callbacks:
- `EvalCallback(..., eval_freq=10000, best_model_save_path=logs/<run>/best, log_path=logs/<run>/eval)`
- `JsonLinesMetricCallback(artifacts/<run>/metrics.jsonl, log_every_steps=10000)`

Writes `artifacts/<run>/summary.json` with paths to the key artifacts.

### 3.5 SAC training (`train_sac_mac.py`)

Defaults:
- `cheetah`: task `run`, run name `sac_cheetah`, steps `10_000`
- `walker`: task `walk`, run name `sac_walker`, steps `10_000`

Model config:
- Policy: `"MlpPolicy"`
- Key SAC params: `buffer_size=500_000`, `batch_size=256`, `tau=0.005`, `gamma=0.99`

Callbacks:
- `EvalCallback(..., eval_freq=20000, ...)`
- `JsonLinesMetricCallback(..., log_every_steps=10000)`

## 4) Offline results: plotting and evaluation

### 4.1 Plotting CLI (`plot_results.py`)

What it does:
- Looks for evaluation traces in either:
  - SB3: `logs/<run>/eval/evaluations.npz`
  - TD‑MPC2-ish formats:
    - `logs/<run>/eval/evaluations.npz`
    - `artifacts/<run>/metrics.jsonl` (using TD‑MPC2 key heuristics)
- Generates overview plots and phase-specific plots under `artifacts/plots/...`.

Notable nuance:
- It supports multiple TD‑MPC2 result formats (`load_tdmpc2` handles `.npz` and a legacy array-like format).
- It has “phase” conventions (Phase 0 baselines, Phase 1 MLP, Phase 2 SSM comparison, Phase 3 horizon ablation).

### 4.2 Phase 4 evaluation (`evaluation/main.py`)

Phase 4 is the “offline report generator”:

- Discovers runs from `artifacts/*/summary.json` (`evaluation/compare_plots.py:discover_runs`).
- Plots:
  - reward curves (Fig 1)
  - sample efficiency bars (Fig 4)
- Builds a `comparison_table.csv` with:
  - `best_eval_mean_reward` from `summary.json` if present (not currently written by PPO/SAC scripts)
  - `final_logged_reward` from metrics series
  - `final_latent_mse` if rollout error series exists
  - `ms_per_world_model_rollout` from `evaluation/benchmark.py` when the model is reconstructable
- Writes a short `report.md` ranking TD‑MPC2 runs by latent MSE.

Important nuance:
- `evaluation/main.py` imports `tdmpc2.compat.load_tdmpc2_agent` which is **missing** in this checkout, so Phase 4 will break if it tries to benchmark TD‑MPC2 models (or even import that module).

### 4.3 Run discovery and time series extraction (`evaluation/compare_plots.py`)

- `discover_runs()` reads each run’s `summary.json`, then loads `metrics.jsonl` rows if the file exists.
- `load_reward_series()` chooses:
  - TD‑MPC2-like reward keys from metrics rows when `algorithm` contains “td-mpc2”
  - else fall back to SB3 `evaluations.npz` if available
  - else fall back to metrics keys again
- `plot_sample_efficiency()` has logic to choose checkpoints:
  - uses fixed checkpoints (50k/100k/200k) if there’s data
  - otherwise derives shared checkpoints from available runs

## 5) Live rollout engine (backend)

### 5.1 Checkpoint registry (`server/config.py`)

`ALGORITHM_REGISTRY` is the “what to load” list for the live server.

Each entry defines:
- `artifact_dir` (used as run name/label and as folder under `artifacts/`)
- `algo_type`: `ppo` | `sac` | `tdmpc`
- `env_name`, `task`
- `checkpoint` filename (SB3 `.zip` vs TD‑MPC2 `.pt`)

Also defines stream/planning constants:
- `STREAM_FPS=20`
- `MPPI_HORIZON=5`
- `MPPI_SAMPLES=256`
- render width/height \(320×240\)

### 5.2 Model loading (`server/model_loader.py`)

Core behavior:
- For each registry entry, it tries multiple checkpoint locations in priority order:
  - `artifacts/<run>/<checkpoint>`
  - `logs/<run>/<checkpoint>`
  - `logs/<run>/best/best_model.*`
  - `logs/<run>/final_model.*` or `final_model`
- Loads:
  - PPO/SAC via SB3 `PPO.load` / `SAC.load`
  - TD‑MPC2 via `load_tdmpc_model(...)` → `tdmpc2.compat.load_tdmpc2_agent(...)` (missing here)
- `load_all_models()` continues on missing checkpoints and prints “Skipping …”.

### 5.3 Rollouts (`server/rollout_engine.py`)

`RolloutEngine` constructs a `ModelAgent` per loaded model dict.

Each `ModelAgent`:
- Creates its own environment via `make_env(..., vectorized=False, render_mode="rgb_array")`.
- Maintains per-agent episodic state:
  - `total_reward`, `step_count`, `done`
  - `low_motion_steps` rescue counter
  - `reset_count`
- Chooses action by algo type:
  - **TD‑MPC2 agent API**: if `algo_type == "tdmpc"` and model has `.act`, it calls:
    - `model.act(obs_tensor, t0=<first step>, eval_mode=True)`
  - **Legacy world-model object**: if `algo_type == "tdmpc"` but no `.act`:
    - builds `planning.MPPI` with the model, encodes obs into latent `z = model.encoder(obs)`
    - plans `action = planner.plan(z, device)`
    - updates latent via `model.encoder(next_obs)` (not via `dynamics` rollout for the live loop)
  - **SB3**: `model.predict(obs, deterministic=True)`

“Collapsed policy rescue”:
- If \(||a||\) stays < `0.075` for 12 steps, it blends in random actions:
  - blend is `0.35` for TD‑MPC2, `0.5` for SB3
- If low motion persists >= 48 steps and `total_reward < 5.0`, it forces episode termination.

Frame rendering:
- Uses `env.render()` which returns an RGB array from `dm_control` physics renderer.

Metrics emitted per step include:
- label/run_name/display_name/algorithm_name
- step, reward, episode_reward, done
- action magnitude and rescue counters
- env_name/task

### 5.4 Video encoding (`server/video_stream.py`)

- Converts RGB→BGR (OpenCV), resizes to \(320×240\), JPEG-encodes, base64 encodes.
- Frontend consumes as `data:image/jpeg;base64,<frame>`.

## 6) Frontend dashboard (Vite + React)

### 6.1 Dev wiring (`dashboard/vite.config.ts`)

The Vite dev server proxies:
- `/ws` → `ws://localhost:8000` (WebSocket, `ws: true`)
- `/metrics`, `/artifacts`, `/health` → `http://localhost:8000`

So the frontend can call relative paths in dev mode without CORS pain.

### 6.2 App layout (`dashboard/src/app/App.tsx`)

High-level UX:
- Header: project title + theme toggle.
- Tabs:
  - **Live Rollout**: video tiles + a live reward chart.
  - **Training Curves**: reward vs steps from JSONL.
  - **Rollout Error**: latent MSE vs horizon.
  - **Results**: comparison table from Phase 4 CSV.
- An “Environment Dial” filter toggles between `all`, `walker`, `cheetah`.

Data sources:
- `useWebSocket()` connects to `/ws` and stores the latest message.
- `useArtifact("/health")` drives:
  - available models
  - error banners (startup/stream errors)
- Other tabs call:
  - `/artifacts/reward-curves`
  - `/artifacts/rollout-errors`
  - `/artifacts/comparison-table`

### 6.3 WebSocket contract (`use-websocket.ts`)

Message shape (TypeScript):
- `labels: string[]`
- `models?: LiveModelCard[]`
- `frames: string[]`
- `metrics: StepMetric[]`

Reconnect behavior:
- If the socket closes, it retries after 2 seconds.

### 6.4 Artifact fetching (`use-artifact.ts`)

Simple `fetch(getApiUrl(path))` with local state:
- `data`, `loading`, `error`

### 6.5 Charts and table components

- `TrainingCurves`: merges per-run series into a shared timestep index, renders multi-line chart.
- `RolloutErrorChart`: renders per-algo error arrays; x-axis is “horizon step”.
- `LiveRewardChart`: maintains an in-memory rolling history (300 points) keyed by model label.
- `ComparisonTable`: renders arbitrary CSV-derived row objects as an HTML table; supports env filtering via heuristics in `inferEnvironment(...)`.
- `VideoPanel`: displays model card metadata + frame image + current step metrics.

## 7) Local SSM dynamics modules (what exists here)

The code under `ssm/` is **not** the full official S4/S5/Mamba implementations; it is a minimal project-local implementation used to provide an SSM-style dynamics module with a compatible interface.

### 7.1 `ssm/ssm_world_model.py` → `SSMDynamics`

`SSMDynamics` is designed to be a **drop-in** replacement for an MLP dynamics module with signature:

```text
z_next = dynamics(z, action)
```

Key features:
- Variant selection:
  - `s5` → `S5Layer`
  - `s4` → `S4Layer`
  - `mamba` → `MambaLayer`
- Maintains a persistent hidden state `self._hidden` for recurrent stepping.
- Exposes:
  - `reset_hidden(batch_size, device)`
  - `snapshot_hidden()` / `restore_hidden(hidden)` to support planning rollouts that need to “branch” hidden state.
- Applies `Dropout(p=0.1)` and then a projection `Linear(state_dim → latent_dim)`.
- Optional `SimNorm` on the output latent.

### 7.2 S5 layer (`ssm/s5_layer.py`)

- Uses a diagonal continuous-time state matrix initialization (`make_hippo_diag`).
- Discretizes via per-dimension \( \Delta t \) to compute:
  - \( \bar{A} = \exp(\Delta t \cdot A) \)
  - \( \bar{B} = \frac{\exp(\Delta t A) - 1}{A} B \) (with safe handling near \(A \approx 0\))
- Provides:
  - `step(z_prev, u_t)` for recurrent stepping
  - sequential forward; “parallel scan” currently falls back to sequential and emits a warning once.

### 7.3 S4 layer (`ssm/s4_layer.py`)

- Implements an S4-style diagonal stable parameterization (simplified):
  - stores `log_neg_a` and `log_dt`
  - clamps to keep parameters in a stable range
  - computes discretized recurrence and applies `step`.

### 7.4 Mamba layer (`ssm/mamba_layer.py`)

- Minimal selective dynamics:
  - \( \Delta t = \mathrm{softplus}(W_{\Delta t} u_t) \)
  - \( \bar{A} = \exp(\Delta t \cdot A) \)
  - \( \bar{B} = (W_B u_t) \odot \Delta t \)
  - \( h_{t+1} = \bar{A} \odot h_t + \bar{B} \)

This captures the “input-dependent” update idea but is not a full Mamba block (no mixing, gating, fused kernels, etc.).

## 8) Planning utilities

### 8.1 MPPI (`planning/mppi.py`)

Implements Model Predictive Path Integral control over a learned world model:

- Samples `N` action sequences of horizon `H` around a nominal sequence.
- Rolls out the world model in latent space:
  - rewards via `model.reward(z, a)`
  - next latent via `model.dynamics(z, a)`
- Uses discounted sum with `gamma`.
- Converts returns to weights with a temperature-scaled softmax.
- Outputs the weighted average **first action**.
- Maintains a shifted nominal action sequence across calls for receding horizon control.

Dropout nuance:
- During planning, it temporarily enables dropout modules (`torch.nn.Dropout`) inside the model so you can do MC-dropout-style sampling if desired.

### 8.2 InfoProp (`planning/info_prop.py`)

Implements “uncertainty-aware rollout truncation”:
- Uses MC dropout disagreement over `K` passes to compute normalized predictive variance.
- If variance exceeds a threshold:
  - trajectory is truncated and optionally bootstrapped through value estimates (if the model supports it).
- This is computationally expensive (\(O(KHN)\)) and biased toward avoiding uncertain regions.

### 8.3 SAM optimizer (`planning/sam_optimizer.py`)

Implements Sharpness-Aware Minimization as an optimizer wrapper requiring a closure with two forward/backward passes.

## 9) Common failure modes (practical)

- **TD‑MPC2 imports fail**: `tdmpc2/` is empty and `tdmpc_2` submodule is missing. Fix by initializing the submodule and making TD‑MPC2 importable.
- **Live server shows “no checkpoints found”**: `server/model_loader.py` only loads what exists on disk. Ensure you have either:
  - `artifacts/<run>/<checkpoint>` OR
  - `logs/<run>/best/best_model.*` OR
  - `logs/<run>/final_model.*`
- **Artifacts overwritten**: `init_run_paths` deletes existing run directories for the same `run_name`.
- **dm_control rendering issues**: headless systems may need proper EGL/OSMesa configuration.

## 10) Extending the project (where to plug in)

- **Add a new algorithm to live dashboard**:
  - Add entry to `server/config.py:ALGORITHM_REGISTRY`
  - Ensure checkpoint discovery paths in `server/model_loader.py` cover your format
  - Ensure `ModelAgent.step()` can produce an action for your model type
- **Add a new chart**:
  - Backend: add a loader endpoint in `server/server.py` and parsing in `server/metrics.py`
  - Frontend: add a tab + `useArtifact(...)` call + chart component
- **Swap dynamics variant**:
  - `SSMDynamics(variant="s4"|"s5"|"mamba")` in the world-model code (currently lives outside this repo; the TD‑MPC2 integration is the expected consumer).

## Appendix A) File-by-file index (entire repo)

This is a one-line purpose map of every source file in this checkout.

### Python (repo root)

- `app.py`: backend + (optional) dashboard dev server launcher; also exports lazy `app` for `uvicorn app:app`.
- `main.py`: unified CLI dispatcher for training, plotting, and evaluation phases.
- `requirements.txt`: Python dependency list.
- `env_setup.py`: `dm_control` → Gymnasium wrapper (`DMCWrapper`) + vectorized env creation for SB3.
- `device_utils.py`: device selection (`cuda`/`mps`/`cpu`) and description string.
- `run_layout.py`: creates `logs/<run>` and `artifacts/<run>`; **deletes existing** dirs of same run name.
- `artifact_logging.py`: JSONL metric callback for SB3 logger snapshots.
- `plot_results.py`: plotting CLI writing into `artifacts/plots/...` from `logs/*/eval` and/or `artifacts/*/metrics.jsonl`.
- `test_env.py`: quick smoke test for env wrapper.
- `train_ppo_mac.py`: PPO baseline training script; writes `artifacts/<run>/summary.json` and `metrics.jsonl`.
- `train_sac_mac.py`: SAC baseline training script; writes `artifacts/<run>/summary.json` and `metrics.jsonl`.
- `train_tdmpc2_s4.py`: thin wrapper calling `tdmpc2.train_tdmpc2` with `dynamics_type="s4"` (requires TD‑MPC2 code).
- `train_tdmpc2_s5.py`: thin wrapper calling `tdmpc2.train_tdmpc2` with `dynamics_type="s5"` (requires TD‑MPC2 code).
- `train_tdmpc2_mamba.py`: thin wrapper calling `tdmpc2.train_tdmpc2` with `dynamics_type="mamba"` (requires TD‑MPC2 code).

### Python (backend: `server/`)

- `server/server.py`: FastAPI app; startup model loading; live rollout loop; WebSocket + REST endpoints; optional static UI mount.
- `server/config.py`: model registry + streaming/planning constants.
- `server/model_loader.py`: finds checkpoints and loads SB3 models or TD‑MPC2 agents (requires TD‑MPC2 code).
- `server/rollout_engine.py`: steps each model in its own env; chooses actions; returns frames + metrics per tick.
- `server/metrics.py`: live rolling window tracker + artifact file loaders for charts/tables.
- `server/video_stream.py`: frame resize + JPEG/base64 encoding for WebSocket payload.
- `server/__init__.py`: package marker.

### Python (evaluation: `evaluation/`)

- `evaluation/main.py`: Phase 4; discovers runs, plots reward curves + sample efficiency, writes comparison CSV + report.
- `evaluation/compare_plots.py`: run discovery (`artifacts/*/summary.json`) and plot helpers; metric key heuristics.
- `evaluation/benchmark.py`: microbenchmark of TD‑MPC2 action/rollout speed for table reporting.
- `evaluation/eval_runner.py`: programmatic evaluation loop with MPPI planning (or model `.act` if present).
- `evaluation/rollout_error.py`: compute and plot multi-step latent rollout error (latent MSE vs horizon).
- `evaluation/__init__.py`: package marker.

### Python (planning: `planning/`)

- `planning/mppi.py`: MPPI planning implementation used in evaluation/live rollouts for world-model objects.
- `planning/info_prop.py`: uncertainty-aware rollout truncation using MC dropout disagreement.
- `planning/sam_optimizer.py`: SAM optimizer wrapper (two-step sharpness-aware updates).
- `planning/sim_norm.py`: “simplex normalization” module used in `SSMDynamics` output.
- `planning/__init__.py`: package marker.

### Python (SSM layers: `ssm/`)

- `ssm/ssm_world_model.py`: `SSMDynamics` (s4/s5/mamba variants) with recurrent hidden state + snapshot/restore.
- `ssm/s4_layer.py`: simplified diagonal S4-style recurrence (`step` + sequential `forward`).
- `ssm/s5_layer.py`: simplified diagonal S5-style recurrence (`step` + sequential `forward`).
- `ssm/mamba_layer.py`: minimal selective SSM recurrence inspired by Mamba.
- `ssm/__init__.py`: package marker.

### Frontend (dashboard: `dashboard/`)

The frontend is a Vite + React app.

- `dashboard/src/main.tsx`: React entrypoint mounting `App`.
- `dashboard/src/app/App.tsx`: main dashboard layout, tabs, and environment filter; drives the app’s data flows.
- `dashboard/src/app/hooks/use-websocket.ts`: connects to `/ws` and parses the live payload.
- `dashboard/src/app/hooks/use-artifact.ts`: fetch helper for REST endpoints.
- `dashboard/src/app/components/video-panel.tsx`: live frame tile + metadata + key metrics.
- `dashboard/src/app/components/live-reward-chart.tsx`: live reward chart with an in-memory rolling history.
- `dashboard/src/app/components/training-curves.tsx`: reward curves chart from `/artifacts/reward-curves`.
- `dashboard/src/app/components/rollout-error-chart.tsx`: rollout error chart from `/artifacts/rollout-errors`.
- `dashboard/src/app/components/comparison-table.tsx`: results table from `/artifacts/comparison-table`.
- `dashboard/src/app/components/theme-provider.tsx` / `theme-toggle.tsx`: dark/light theme wiring.
- `dashboard/src/app/components/ui/*`: UI primitives (buttons, dialogs, tabs, etc.) used by the dashboard.


