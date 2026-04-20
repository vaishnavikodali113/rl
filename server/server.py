import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from device_utils import get_best_device
from server.config import BASE_DIR, STREAM_FPS, MPPI_HORIZON, MPPI_SAMPLES
from server.model_loader import load_all_models
from server.rollout_engine import RolloutEngine
from server.metrics import MetricsTracker
from server.video_stream import encode_all_frames

# ── Global state ─────────────────────────────────────────────────────────────
engine: RolloutEngine | None = None
tracker: MetricsTracker | None = None
clients: set[WebSocket] = set()
stream_task: asyncio.Task | None = None
latest_payload: dict | None = None
stream_error: str | None = None
startup_error: str | None = None


async def broadcast_payload(payload: dict):
    stale_clients: list[WebSocket] = []

    for client in list(clients):
        try:
            await client.send_json(payload)
        except Exception:
            stale_clients.append(client)

    for client in stale_clients:
        clients.discard(client)


async def rollout_loop():
    global latest_payload, stream_error

    frame_interval = 1.0 / STREAM_FPS

    while True:
        t0 = time.perf_counter()

        if engine is None or tracker is None or not clients:
            await asyncio.sleep(frame_interval)
            continue

        try:
            frames, metrics = engine.run_step()
            tracker.update(metrics)
            encoded_frames = encode_all_frames(frames)
            latest_payload = {
                "labels": engine.labels,
                "models": engine.model_cards,
                "frames": encoded_frames,
                "metrics": metrics,
            }
            stream_error = None
        except Exception as exc:
            stream_error = f"{type(exc).__name__}: {exc}"
            print(f"[server] Live rollout error: {stream_error}")
            await asyncio.sleep(frame_interval)
            continue

        await broadcast_payload(latest_payload)

        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, frame_interval - elapsed))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tracker, stream_task, startup_error
    
    device = get_best_device()
    print(f"[server] Loading models on device: {device}")

    try:
        models = load_all_models(device=str(device))
        engine = RolloutEngine(models, device=str(device),
                               mppi_horizon=MPPI_HORIZON, mppi_samples=MPPI_SAMPLES)
        tracker = MetricsTracker()
        startup_error = None
        print(f"[server] Ready — streaming {len(models)} model(s)")
        stream_task = asyncio.create_task(rollout_loop())
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        engine = None
        tracker = None
        print(f"[server] Startup error: {startup_error}")
    yield
    if stream_task is not None:
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="RL Visualization API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket: live rollout stream ────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    
    try:
        if latest_payload is not None:
            await ws.send_json(latest_payload)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        print("[server] Client disconnected")
    finally:
        clients.discard(ws)

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
    """Final comparison table from artifacts/evaluation_pngs/comparison_table.csv."""
    return MetricsTracker.load_comparison_table()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": engine.model_cards if engine else [],
        "startup_error": startup_error,
        "stream_error": stream_error,
    }

# ── Static Frontend ──────────────────────────────────────────────────────────
# Mount the compiled Vite dashboard so it is served on port 8000 alongside the API
frontend_dist = os.path.join(BASE_DIR, "dashboard", "dist")
if os.path.isdir(frontend_dist):
    # html=True means it serves index.html at exactly "/"
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    print(f"[server] Warning: Frontend build not found at {frontend_dist}.")
    print("[server] To serve the UI from here, please run 'npm run build' inside the 'dashboard' directory first.")
