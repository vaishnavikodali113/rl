from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime, timezone


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    log_dir: Path
    best_dir: Path
    eval_dir: Path
    artifact_dir: Path
    metrics_path: Path
    summary_path: Path
    model_path: Path
    eval_npz_path: Path


def init_run_paths(run_name: str) -> RunPaths:
    log_dir = Path("logs") / run_name
    artifact_dir = Path("artifacts") / run_name
    best_dir = log_dir / "best"
    eval_dir = log_dir / "eval"

    backups = []
    try:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        if log_dir.exists():
            log_bkp = log_dir.with_name(f"{log_dir.name}_backup_{timestamp}")
            log_dir.rename(log_bkp)
            backups.append((log_bkp, log_dir))
        if artifact_dir.exists():
            art_bkp = artifact_dir.with_name(f"{artifact_dir.name}_backup_{timestamp}")
            artifact_dir.rename(art_bkp)
            backups.append((art_bkp, artifact_dir))
    except Exception as e:
        for bkp, orig in backups:
            if bkp.exists():
                bkp.rename(orig)
        raise RuntimeError(f"Failed to safely backup existing run directories for {run_name}") from e

    # Cleanup old backups (keep latest 5)
    for base_path, name in [(Path("logs"), log_dir.name), (Path("artifacts"), artifact_dir.name)]:
        if base_path.exists():
            old_bkps = sorted(base_path.glob(f"{name}_backup_*"))
            for old in old_bkps[:-5]:
                shutil.rmtree(old, ignore_errors=True)

    for path in (log_dir, best_dir, eval_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        log_dir=log_dir,
        best_dir=best_dir,
        eval_dir=eval_dir,
        artifact_dir=artifact_dir,
        metrics_path=artifact_dir / "metrics.jsonl",
        summary_path=artifact_dir / "summary.json",
        model_path=log_dir / "final_model",
        eval_npz_path=eval_dir / "evaluations.npz",
    )
