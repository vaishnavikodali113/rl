from pathlib import Path

from server.config import ARTIFACTS_DIR, ALGORITHM_REGISTRY, BASE_DIR
from tdmpc2.compat import load_tdmpc2_agent

LOGS_DIR = Path(BASE_DIR) / "logs"


def _checkpoint_candidates(cfg: dict) -> list[Path]:
    artifact_dir = cfg["artifact_dir"]
    checkpoint_name = cfg["checkpoint"]
    algo = cfg["algo_type"]

    candidates = [Path(ARTIFACTS_DIR) / artifact_dir / checkpoint_name]

    if algo in {"ppo", "sac"}:
        candidates.extend(
            [
                LOGS_DIR / artifact_dir / checkpoint_name,
                LOGS_DIR / artifact_dir / "best" / "best_model.zip",
                LOGS_DIR / artifact_dir / "final_model.zip",
                LOGS_DIR / artifact_dir / "final_model",
            ]
        )
    elif algo == "tdmpc":
        candidates.extend(
            [
                LOGS_DIR / artifact_dir / checkpoint_name,
                LOGS_DIR / artifact_dir / "best" / "best_model.pt",
                LOGS_DIR / artifact_dir / "final_model.pt",
                LOGS_DIR / artifact_dir / "final_model",
            ]
        )

    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _resolve_checkpoint_path(cfg: dict) -> str:
    for candidate in _checkpoint_candidates(cfg):
        if candidate.is_file():
            return str(candidate)
    checked = ", ".join(str(path) for path in _checkpoint_candidates(cfg))
    raise FileNotFoundError(
        f"No checkpoint found for {cfg['artifact_dir']}. Checked: {checked}"
    )


def _env_title(env_name: str, task: str) -> str:
    return f"{env_name.title()} {task.title()}"


def _env_theme(env_name: str) -> str:
    return "speed" if env_name == "cheetah" else "balance"


def _behavior_text(env_name: str, task: str) -> str:
    verb_map = {
        "walk": "walking",
        "run": "running",
        "hop": "hopping",
    }
    return f"{env_name.title()} {verb_map.get(task, task)}"


def _preferred_dynamics_type(run_name: str) -> str:
    if "mamba" in run_name:
        return "mamba"
    if "s5" in run_name:
        return "s5"
    if "s4" in run_name:
        return "s4"
    return "mlp"


def _summary_candidates_for_checkpoint(checkpoint_path: Path) -> list[Path]:
    return [
        checkpoint_path.parent / "summary.json",
        checkpoint_path.parent.parent / "summary.json",
        checkpoint_path.parent.parent.parent / "summary.json",
        Path(ARTIFACTS_DIR) / checkpoint_path.stem / "summary.json",
        Path(ARTIFACTS_DIR) / checkpoint_path.parent.name / "summary.json",
    ]


def load_tdmpc_model(ckpt_path: str, device: str, preferred_dynamics_type: str):
    """
    Loads a TD-MPC2 agent from the compatibility artifacts created by tdmpc2/compat.py.
    """
    checkpoint_path = Path(ckpt_path)
    summary_path = next(
        (candidate for candidate in _summary_candidates_for_checkpoint(checkpoint_path) if candidate.is_file()),
        None,
    )
    if summary_path is None:
        raise FileNotFoundError(
            f"Missing TD-MPC2 summary.json for checkpoint {ckpt_path}."
        )
    return load_tdmpc2_agent(
        checkpoint_path,
        summary_path=summary_path,
        device=device,
        dynamics_type=preferred_dynamics_type,
    )


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
    checkpoint_path = _resolve_checkpoint_path(cfg)
    env_title = _env_title(cfg["env_name"], cfg["task"])

    if algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint_path, device=device)
    elif algo == "sac":
        from stable_baselines3 import SAC

        model = SAC.load(checkpoint_path, device=device)
    elif algo == "tdmpc":
        model = load_tdmpc_model(
            checkpoint_path,
            device,
            preferred_dynamics_type=_preferred_dynamics_type(artifact_dir),
        )
    else:
        raise ValueError(f"Unknown algo type: {algo}")

    return {
        "model": model,
        "algo_type": algo,
        "env_name": cfg["env_name"],
        "task": cfg["task"],
        "env_theme": _env_theme(cfg["env_name"]),
        "label": artifact_dir,
        "run_name": artifact_dir,
        "display_name": artifact_dir,
        "algorithm_name": label,
        "env_title": env_title,
        "behavior_text": _behavior_text(cfg["env_name"], cfg["task"]),
        "checkpoint_path": checkpoint_path,
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
        except Exception as e:
            print(f"[model_loader] Error loading {label}: {e}")
            
    return loaded
