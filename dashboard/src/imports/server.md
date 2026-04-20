# RL Visualization Backend — `server.md`

> **Role:** FastAPI + WebSocket server that loads trained RL models, runs synchronized evaluation rollouts, and streams live video frames + metrics to the Next.js dashboard.
> **Reads from:** `artifacts/*/metrics.jsonl`, `artifacts/*/rollout_errors.npy`, `artifacts/comparison_table.csv`
> **Serves:** WebSocket on `ws://localhost:8000/ws`, REST API on `http://localhost:8000`

---

## Where This Lives in the Project

Place the entire `server/` directory at the project root, alongside `tdmpc2/`, `ssm/`, `planning/`, and `evaluation/`:

```
rl/
├── server/
│   ├── __init__.py
│   ├── server.py              ← FastAPI app + WebSocket endpoint
│   ├── model_loader.py        ← loads PPO / SAC / TD-MPC2 checkpoints
│   ├── rollout_engine.py      ← runs synchronized multi-model rollouts
│   ├── metrics.py             ← tracks and aggregates live metrics
│   ├── video_stream.py        ← encodes rendered frames to base64 JPEG
│   └── config.py              ← paths, FPS, algorithm registry
│
├── artifacts/                 ← all training outputs live here (your setup)
│   ├── ppo_walker/
│   │   └── metrics.jsonl
│   ├── sac_cheetah/
│   │   └── metrics.jsonl
│   ├── tdmpc2_walker_mlp/
│   │   ├── metrics.jsonl
│   │   └── rollout_errors.npy
│   ├── tdmpc2_walker_s4/
│   │   ├── metrics.jsonl
│   │   └── rollout_errors.npy
│   ├── tdmpc2_walker_s5/
│   │   ├── metrics.jsonl
│   │   └── rollout_errors.npy
│   ├── tdmpc2_walker_mamba/
│   │   ├── metrics.jsonl
│   │   └── rollout_errors.npy
│   ├── fig1_reward_curves.png
│   ├── fig2_rollout_error.png
│   ├── fig3_planning_stability.png
│   ├── fig4_sample_efficiency.png
│   └── comparison_table.csv
│
├── env_setup.py
├── device_utils.py
├── planning/
│   ├── mppi.py
│   └── info_prop.py
└── evaluation/
    └── rollout_error.py
```

> **Note on your artifact path:** All Phase 4 outputs (`fig*.png`, `comparison_table.csv`, `rollout_errors.npy`) are under `artifacts/` in your setup. Every path reference in this server reflects that — there are no `logs/` reads.

---

## Architecture

```
RL Checkpoints  (artifacts/*/  ← PPO .zip, SAC .zip, TD-MPC2 .pt)
        ↓
model_loader.py     (loads correct model class per algo type)
        ↓
rollout_engine.py   (steps all models in sync, captures frames)
        ↓
video_stream.py     (renders → base64 JPEG per frame)
metrics.py          (accumulates reward, uncertainty, action magnitude)
        ↓
server.py           (FastAPI — WebSocket stream + REST endpoints)
        ↓
ws://localhost:8000/ws   →   Next.js Dashboard
http://localhost:8000/metrics/history
http://localhost:8000/artifacts/reward-curves
http://localhost:8000/artifacts/rollout-errors
http://localhost:8000/artifacts/comparison-table
```

---

## Step 1 — `server/config.py`

Central registry for model paths, environment names, and stream settings.

```python
# server/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Registry of all trained algorithms.
# Each entry: display label → (artifact subfolder, algo type, env_name, task)
ALGORITHM_REGISTRY = {
    "PPO (Walker)": {
        "artifact_dir": "ppo_walker",
        "algo_type": "ppo",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.zip",      # SB3 saves as .zip
    },
    "SAC (Cheetah)": {
        "artifact_dir": "sac_cheetah",
        "algo_type": "sac",
        "env_name": "cheetah",
        "task": "run",
        "checkpoint": "model.zip",
    },
    "TD-MPC2 MLP": {
        "artifact_dir": "tdmpc2_walker_mlp",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 S4": {
        "artifact_dir": "tdmpc2_walker_s4",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 S5": {
        "artifact_dir": "tdmpc2_walker_s5",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
    "TD-MPC2 Mamba": {
        "artifact_dir": "tdmpc2_walker_mamba",
        "algo_type": "tdmpc",
        "env_name": "walker",
        "task": "walk",
        "checkpoint": "model.pt",
    },
}

STREAM_FPS = 20          # WebSocket target frame rate
MPPI_HORIZON = 5         # Planning horizon used during live rollout
MPPI_SAMPLES = 256       # Reduced vs. training for real-time speed
RENDER_WIDTH = 320
RENDER_HEIGHT = 240
```

---

## Step 2 — `server/model_loader.py`

Loads the correct model class for each algorithm type and returns a unified `predict(obs_tensor, z=None)` interface.

```python
# server/model_loader.py
import os
import torch
from stable_baselines3 import PPO, SAC
from server.config import ARTIFACTS_DIR, ALGORITHM_REGISTRY


def load_tdmpc_model(artifact_dir: str, device: str):
    """
    Loads a TD-MPC2 model (.pt checkpoint).
    Expects the checkpoint to contain {"model_state": ..., "obs_dim": ..., "act_dim": ..., "dynamics_type": ...}
    """
    from tdmpc2.model import TDMPC2Model
    from ssm.ssm_world_model import SSMWorldModel  # Phase 2 drop-in

    ckpt_path = os.path.join(ARTIFACTS_DIR, artifact_dir, "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    dynamics_type = ckpt.get("dynamics_type", "mlp")   # "mlp" | "s4" | "s5" | "mamba"
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]

    model = TDMPC2Model(obs_dim, act_dim, dynamics_type=dynamics_type)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def load_model(label: str, device: str = "cpu"):
    """
    Returns a dict with keys:
        "model"     — the loaded model object
        "algo_type" — "ppo" | "sac" | "tdmpc"
        "env_name"  — environment string
        "task"      — task string
    """
    cfg = ALGORITHM_REGISTRY[label]
    algo = cfg["algo_type"]
    artifact_dir = cfg["artifact_dir"]

    if algo == "ppo":
        ckpt_path = os.path.join(ARTIFACTS_DIR, artifact_dir, cfg["checkpoint"])
        model = PPO.load(ckpt_path, device=device)

    elif algo == "sac":
        ckpt_path = os.path.join(ARTIFACTS_DIR, artifact_dir, cfg["checkpoint"])
        model = SAC.load(ckpt_path, device=device)

    elif algo == "tdmpc":
        model = load_tdmpc_model(artifact_dir, device)

    else:
        raise ValueError(f"Unknown algo type: {algo}")

    return {
        "model": model,
        "algo_type": algo,
        "env_name": cfg["env_name"],
        "task": cfg["task"],
        "label": label,
    }


def load_all_models(device: str = "cpu") -> list[dict]:
    """Load every algorithm in the registry. Skip missing checkpoints gracefully."""
    loaded = []
    for label in ALGORITHM_REGISTRY:
        try:
            loaded.append(load_model(label, device))
            print(f"[model_loader] Loaded: {label}")
        except FileNotFoundError:
            print(f"[model_loader] Skipping {label} — checkpoint not found")
    return loaded
```

---

## Step 3 — `server/rollout_engine.py`

Runs all models in lockstep. Each call to `run_step()` advances every model's environment by one timestep and returns rendered frames and per-model metrics.

```python
# server/rollout_engine.py
import numpy as np
import torch
from env_setup import make_env
from planning.mppi import MPPI


class ModelAgent:
    """Wraps a single model + env pair for uniform step() interface."""

    def __init__(self, model_dict: dict, device: str, mppi_horizon: int, mppi_samples: int):
        self.label = model_dict["label"]
        self.algo_type = model_dict["algo_type"]
        self.model = model_dict["model"]
        self.device = device

        self.env = make_env(model_dict["env_name"], model_dict["task"])
        self.obs, _ = self.env.reset()
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False

        # MPPI planner only for TD-MPC2 models
        if self.algo_type == "tdmpc":
            act_dim = self.env.action_space.shape[0]
            self.planner = MPPI(
                self.model,
                act_dim,
                horizon=mppi_horizon,
                n_samples=mppi_samples,
            )
            # Encode initial observation
            obs_t = torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0)
            self.z = self.model.encoder(obs_t)
        else:
            self.planner = None
            self.z = None

    def step(self):
        """Advance environment by one step. Returns (frame, step_metrics)."""
        if self.done:
            self.obs, _ = self.env.reset()
            self.total_reward = 0.0
            self.step_count = 0
            self.done = False
            if self.algo_type == "tdmpc":
                obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.z = self.model.encoder(obs_t)

        # Select action
        if self.algo_type == "tdmpc":
            with torch.no_grad():
                action = self.planner.plan(self.z, self.device).squeeze(0).cpu().numpy()
        else:
            # SB3 models (PPO / SAC)
            action, _ = self.model.predict(self.obs, deterministic=True)

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        self.total_reward += float(reward)
        self.step_count += 1
        self.obs = next_obs

        # Update latent for TD-MPC2
        if self.algo_type == "tdmpc":
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                self.z = self.model.encoder(obs_t)

        # Render frame (RGB array)
        frame = self.env.render()

        metrics = {
            "label": self.label,
            "step": self.step_count,
            "reward": float(reward),
            "episode_reward": self.total_reward,
            "done": self.done,
            "action_magnitude": float(np.linalg.norm(action)),
        }

        return frame, metrics


class RolloutEngine:
    """Synchronously steps all agents and returns frames + metrics."""

    def __init__(self, model_dicts: list[dict], device: str,
                 mppi_horizon: int = 5, mppi_samples: int = 256):
        self.agents = [
            ModelAgent(md, device, mppi_horizon, mppi_samples)
            for md in model_dicts
        ]

    def run_step(self):
        """Returns: frames (list of np.ndarray), metrics (list of dict)."""
        frames, metrics = [], []
        for agent in self.agents:
            frame, m = agent.step()
            frames.append(frame)
            metrics.append(m)
        return frames, metrics

    @property
    def labels(self):
        return [a.label for a in self.agents]
```

---

## Step 4 — `server/metrics.py`

Accumulates per-step metrics for the live chart data sent to the frontend.

```python
# server/metrics.py
from collections import defaultdict, deque
import json
import os
from server.config import ARTIFACTS_DIR


class MetricsTracker:
    """
    Keeps a rolling window of live step metrics (for WebSocket stream)
    and can load static training history from artifacts/ for the charts.
    """

    def __init__(self, window: int = 500):
        # Live step history per label, bounded to `window` steps
        self.live: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, step_metrics: list[dict]):
        """Called every WebSocket frame with the list of per-model metrics."""
        for m in step_metrics:
            self.live[m["label"]].append({
                "step": m["step"],
                "reward": m["reward"],
                "episode_reward": m["episode_reward"],
                "action_magnitude": m["action_magnitude"],
            })

    def get_live_snapshot(self) -> dict:
        """Returns the current rolling window for all labels."""
        return {label: list(data) for label, data in self.live.items()}

    # ── Static artifact loaders ─────────────────────────────────────────────

    @staticmethod
    def load_training_curves() -> dict:
        """
        Reads artifacts/*/metrics.jsonl and returns timestep + reward arrays.
        Used by the /artifacts/reward-curves REST endpoint.
        """
        result = {}
        for subdir in os.listdir(ARTIFACTS_DIR):
            jsonl_path = os.path.join(ARTIFACTS_DIR, subdir, "metrics.jsonl")
            if not os.path.isfile(jsonl_path):
                continue
            timesteps, rewards = [], []
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        t = d.get("timesteps")
                        r = d.get("metrics", {})
                        reward = r.get("eval/mean_reward") or r.get("rollout/ep_rew_mean")
                        if t is not None and reward is not None:
                            timesteps.append(t)
                            rewards.append(reward)
                    except json.JSONDecodeError:
                        continue
            if timesteps:
                result[subdir] = {"timesteps": timesteps, "rewards": rewards}
        return result

    @staticmethod
    def load_rollout_errors() -> dict:
        """
        Reads artifacts/*/rollout_errors.npy (shape [max_horizon]).
        Used by the /artifacts/rollout-errors REST endpoint.
        """
        result = {}
        for subdir in os.listdir(ARTIFACTS_DIR):
            npy_path = os.path.join(ARTIFACTS_DIR, subdir, "rollout_errors.npy")
            if not os.path.isfile(npy_path):
                continue
            import numpy as np
            errors = np.load(npy_path).tolist()
            result[subdir] = errors
        return result

    @staticmethod
    def load_comparison_table() -> list[dict]:
        """
        Reads artifacts/comparison_table.csv.
        Used by the /artifacts/comparison-table REST endpoint.
        """
        import csv
        path = os.path.join(ARTIFACTS_DIR, "comparison_table.csv")
        if not os.path.isfile(path):
            return []
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
```

---

## Step 5 — `server/video_stream.py`

Encodes NumPy frames to base64 JPEG strings for transport over WebSocket JSON.

```python
# server/video_stream.py
import cv2
import base64
import numpy as np
from server.config import RENDER_WIDTH, RENDER_HEIGHT


def encode_frame(frame: np.ndarray, quality: int = 75) -> str:
    """
    frame: H×W×3 uint8 RGB array (from dm_control render)
    Returns: base64-encoded JPEG string
    """
    # dm_control returns RGB; OpenCV expects BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", bgr, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def encode_all_frames(frames: list[np.ndarray]) -> list[str]:
    return [encode_frame(f) for f in frames]
```

---

## Step 6 — `server/server.py`

The FastAPI application. Exposes one WebSocket endpoint for the live stream and four REST endpoints for static artifact data.

```python
# server/server.py
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from device_utils import get_best_device
from server.config import STREAM_FPS, MPPI_HORIZON, MPPI_SAMPLES
from server.model_loader import load_all_models
from server.rollout_engine import RolloutEngine
from server.metrics import MetricsTracker
from server.video_stream import encode_all_frames

# ── Global state ─────────────────────────────────────────────────────────────
engine: RolloutEngine | None = None
tracker: MetricsTracker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tracker
    device = get_best_device()
    print(f"[server] Loading models on device: {device}")
    models = load_all_models(device=str(device))
    engine = RolloutEngine(models, device=str(device),
                           mppi_horizon=MPPI_HORIZON, mppi_samples=MPPI_SAMPLES)
    tracker = MetricsTracker()
    print(f"[server] Ready — streaming {len(models)} model(s)")
    yield


app = FastAPI(title="RL Visualization API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket: live rollout stream ────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    frame_interval = 1.0 / STREAM_FPS
    try:
        while True:
            t0 = time.perf_counter()
            frames, metrics = engine.run_step()
            tracker.update(metrics)
            encoded_frames = encode_all_frames(frames)
            await ws.send_json({
                "labels": engine.labels,
                "frames": encoded_frames,
                "metrics": metrics,
            })
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, frame_interval - elapsed))
    except WebSocketDisconnect:
        print("[server] Client disconnected")


# ── REST: static artifact endpoints ──────────────────────────────────────────
@app.get("/metrics/live")
async def live_metrics():
    """Latest rolling window of live step metrics (last N steps per model)."""
    return tracker.get_live_snapshot()


@app.get("/artifacts/reward-curves")
async def reward_curves():
    """Training reward curves loaded from artifacts/*/metrics.jsonl."""
    return MetricsTracker.load_training_curves()


@app.get("/artifacts/rollout-errors")
async def rollout_errors():
    """Per-horizon MSE loaded from artifacts/*/rollout_errors.npy."""
    return MetricsTracker.load_rollout_errors()


@app.get("/artifacts/comparison-table")
async def comparison_table():
    """Final comparison table from artifacts/comparison_table.csv."""
    return MetricsTracker.load_comparison_table()


@app.get("/health")
async def health():
    return {"status": "ok", "models": engine.labels if engine else []}
```

---

## Step 7 — Dependencies

Add to `requirements.txt` (if not already present):

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
opencv-python>=4.9.0
python-multipart>=0.0.9
```

---

## Step 8 — Running the Server

From the project root (`rl/`):

```bash
uvicorn server.server:app --reload --host 0.0.0.0 --port 8000
```

The server is ready when you see:

```
[server] Loaded: PPO (Walker)
[server] Loaded: TD-MPC2 MLP
...
[server] Ready — streaming N model(s)
INFO:     Application startup complete.
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Live stream — frames + step metrics at `STREAM_FPS` |
| `/metrics/live` | GET | Rolling window of live step data (JSON) |
| `/artifacts/reward-curves` | GET | Training reward curves from all `metrics.jsonl` files |
| `/artifacts/rollout-errors` | GET | Per-horizon MSE from all `rollout_errors.npy` files |
| `/artifacts/comparison-table` | GET | Rows from `comparison_table.csv` |
| `/health` | GET | Server status + list of loaded model labels |

WebSocket message format (sent at each frame):

```json
{
  "labels": ["PPO (Walker)", "TD-MPC2 MLP", "TD-MPC2 S5"],
  "frames": ["<base64jpg>", "<base64jpg>", "<base64jpg>"],
  "metrics": [
    { "label": "PPO (Walker)", "step": 42, "reward": 1.2, "episode_reward": 38.4, "done": false, "action_magnitude": 0.81 },
    { "label": "TD-MPC2 MLP", "step": 42, "reward": 1.5, "episode_reward": 43.1, "done": false, "action_magnitude": 0.74 },
    { "label": "TD-MPC2 S5",  "step": 42, "reward": 1.7, "episode_reward": 47.2, "done": false, "action_magnitude": 0.69 }
  ]
}
```

---

## Notes

- The server gracefully skips any algorithm whose checkpoint file is missing at startup — you don't need all six models trained before running.
- `device_utils.get_best_device()` from your existing codebase auto-selects CUDA → MPS → CPU, so the server works on both your macOS and Fedora machines without changes.
- For cross-machine development, set `allow_origins=["*"]` in the CORS middleware temporarily.
- dm_control's `render()` requires a display; on headless Fedora Linux run `Xvfb :99 -screen 0 1024x768x24 &` and `export DISPLAY=:99` before starting the server.
