from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"
FRONTEND_DIST_DIR = DASHBOARD_DIR / "dist"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def load_server_app():
    try:
        from server.server import app as server_app
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency while starting the backend: "
            f"{exc.name}\n"
            "Install the project dependencies in your active environment, for example:\n"
            "  pip install -r requirements.txt"
        ) from exc

    return server_app


class LazyASGIApp:
    """Delay importing the full server stack until the app is actually used."""

    def __init__(self) -> None:
        self._app: Any | None = None

    def _get_app(self):
        if self._app is None:
            self._app = load_server_app()
        return self._app

    async def __call__(self, scope, receive, send) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


# Export the FastAPI app at the repo root so both of these work:
#   python app.py serve
#   uvicorn app:app --reload
app = LazyASGIApp()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the RL visualization backend and dashboard."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "status"],
        help="Action to run (default: serve).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Backend host for FastAPI/uvicorn.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend port for FastAPI/uvicorn.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload.",
    )
    parser.add_argument(
        "--frontend",
        choices=["auto", "dev", "static", "none"],
        default="auto",
        help="How to run the dashboard: auto, dev, static, or none.",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=5173,
        help="Port for the Vite dev server when --frontend=dev.",
    )
    parser.add_argument(
        "--public-host",
        default=None,
        help="Browser-facing host used when wiring frontend URLs in dev mode.",
    )
    return parser


def discover_artifacts() -> list[Path]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    return sorted(path for path in ARTIFACTS_DIR.iterdir() if path.is_dir())


def print_status() -> None:
    artifacts = discover_artifacts()
    print(f"Project root: {ROOT_DIR}")
    print(f"Artifacts dir: {ARTIFACTS_DIR} ({'present' if ARTIFACTS_DIR.exists() else 'missing'})")
    print(
        f"Dashboard source: {DASHBOARD_DIR} ({'present' if DASHBOARD_DIR.exists() else 'missing'})"
    )
    print(
        f"Dashboard build: {FRONTEND_DIST_DIR} "
        f"({'present' if FRONTEND_DIST_DIR.exists() else 'missing'})"
    )
    print(f"Artifact runs found: {len(artifacts)}")

    for artifact_dir in artifacts:
        interesting_files = [
            name
            for name in ("metrics.jsonl", "rollout_errors.npy", "summary.json", "comparison_table.csv")
            if (artifact_dir / name).exists()
        ]
        summary = ", ".join(interesting_files) if interesting_files else "no tracked artifact files"
        print(f"  - {artifact_dir.name}: {summary}")


def resolve_public_host(host: str, explicit_public_host: str | None) -> str:
    if explicit_public_host:
        return explicit_public_host
    if host in {"0.0.0.0", "::"}:
        return "localhost"
    return host


def resolve_frontend_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if FRONTEND_DIST_DIR.is_dir():
        return "static"
    if DASHBOARD_DIR.is_dir() and shutil.which("npm"):
        return "dev"
    return "none"


def start_frontend_dev_server(
    backend_port: int,
    frontend_port: int,
    public_host: str,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = f"http://{public_host}:{backend_port}"
    env["VITE_WS_BASE_URL"] = f"ws://{public_host}:{backend_port}"

    cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(frontend_port),
    ]

    print(f"[app] Starting dashboard dev server on http://{public_host}:{frontend_port}")
    return subprocess.Popen(cmd, cwd=DASHBOARD_DIR, env=env, text=True)


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def describe_port_conflict(port: int) -> str:
    if shutil.which("ss") is None:
        return f"Port {port} is already in use."

    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return f"Port {port} is already in use."

    details = result.stdout.strip()
    if not details:
        return f"Port {port} is already in use."

    return (
        f"Port {port} is already in use.\n"
        f"{details}\n"
        f"Stop the existing listener or run with --port {port + 1}."
    )


def ensure_port_available(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    bind_host = "::" if host == "::" else ("0.0.0.0" if host == "0.0.0.0" else host)

    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            raise SystemExit(describe_port_conflict(port))


def serve(args: argparse.Namespace) -> None:
    frontend_mode = resolve_frontend_mode(args.frontend)
    frontend_process: subprocess.Popen[str] | None = None
    public_host = resolve_public_host(args.host, args.public_host)

    ensure_port_available(args.host, args.port)

    if frontend_mode == "dev":
        if not DASHBOARD_DIR.is_dir():
            raise SystemExit("Dashboard directory not found; cannot start frontend dev server.")
        if shutil.which("npm") is None:
            raise SystemExit("npm is not installed; cannot start frontend dev server.")
        frontend_process = start_frontend_dev_server(
            backend_port=args.port,
            frontend_port=args.frontend_port,
            public_host=public_host,
        )
        print(
            f"[app] Backend API: http://{public_host}:{args.port} | "
            f"Frontend UI: http://{public_host}:{args.frontend_port}"
        )
    elif frontend_mode == "static":
        print(f"[app] Serving built dashboard from {FRONTEND_DIST_DIR}")
        print(f"[app] Unified UI and API: http://{public_host}:{args.port}")
    else:
        print(f"[app] Starting backend without a frontend runner (mode={frontend_mode})")
        print(f"[app] API available at http://{public_host}:{args.port}")

    try:
        uvicorn_target = "app:app" if args.reload else load_server_app()
        uvicorn.run(uvicorn_target, host=args.host, port=args.port, reload=args.reload)
    finally:
        stop_process(frontend_process)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        print_status()
        return

    serve(args)


if __name__ == "__main__":
    main()
