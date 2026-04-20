from pathlib import Path

import torch

from server.config import ARTIFACTS_DIR, ALGORITHM_REGISTRY, BASE_DIR

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


def load_tdmpc_model(ckpt_path: str, device: str, preferred_dynamics_type: str):
    """
    Loads a TD-MPC2 model from the trainer checkpoint schema saved in this repo.
    """
    from tdmpc2.model import TDMPC2Model

    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt["model_state_dict"]
    config = ckpt.get("config", {})

    dynamics_type = preferred_dynamics_type

    latent_dim = int(
        config.get("latent_dim", model_state["reward.net.0.weight"].shape[1] - 6)
    )
    obs_dim = int(model_state["encoder.net.0.weight"].shape[1])
    act_dim = int(model_state["reward.net.0.weight"].shape[1] - latent_dim)
    ssm_state_dim = int(config.get("ssm_state_dim", 256))
    simnorm_dim = int(config.get("simnorm_dim", 8))
    if simnorm_dim <= 0:
        simnorm_dim = None

    model = TDMPC2Model(
        obs_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=latent_dim,
        dynamics_type=dynamics_type,
        ssm_state_dim=ssm_state_dim,
        simnorm_dim=simnorm_dim,
    )
    model.load_state_dict(model_state, strict=False)
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
