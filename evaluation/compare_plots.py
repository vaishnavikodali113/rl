from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TDMPC2_REWARD_KEYS = (
    "eval/mean_reward",
    "rollout/ep_rew_mean",
    "rollout/recent_episode_return",
)
ROLLOUT_ERROR_KEYS = ("rollout_error/horizon", "eval/rollout_error_horizon")


def _load_json(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_metrics_rows(path: str | os.PathLike[str]) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(artifacts_root: str | os.PathLike[str] = "artifacts") -> dict[str, dict]:
    runs = {}
    for summary_path in sorted(Path(artifacts_root).glob("*/summary.json")):
        summary = _load_json(summary_path)
        run_name = summary["run_name"]
        artifacts = summary.get("artifacts", {})
        metrics_path = artifacts.get("metrics_jsonl")
        eval_npz_path = artifacts.get("eval_npz")
        metrics_rows = _load_metrics_rows(metrics_path) if metrics_path and os.path.exists(metrics_path) else []
        runs[run_name] = {
            "run_name": run_name,
            "algorithm": summary.get("algorithm", run_name),
            "environment": summary.get("environment"),
            "summary": summary,
            "metrics_path": metrics_path,
            "metrics_rows": metrics_rows,
            "eval_npz_path": eval_npz_path,
        }
    return runs


def _metric_series(metrics_rows: list[dict], keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    timesteps = []
    values = []
    for row in metrics_rows:
        metrics = row.get("metrics", {})
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                timesteps.append(row["timesteps"])
                values.append(float(value))
                break
    return np.asarray(timesteps), np.asarray(values)


def load_reward_series(run_record: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    summary = run_record["summary"]
    algorithm = str(summary.get("algorithm", "")).lower()

    if "td-mpc2" in algorithm:
        timesteps, rewards = _metric_series(run_record["metrics_rows"], TDMPC2_REWARD_KEYS)
        if len(timesteps):
            return timesteps, rewards

    eval_npz_path = run_record.get("eval_npz_path")
    if eval_npz_path and os.path.exists(eval_npz_path):
        data = np.load(eval_npz_path)
        if "timesteps" in data and "results" in data:
            return np.asarray(data["timesteps"]), np.asarray(data["results"]).mean(axis=-1)

    timesteps, rewards = _metric_series(run_record["metrics_rows"], TDMPC2_REWARD_KEYS)
    if len(timesteps):
        return timesteps, rewards
    return None, None


def load_rollout_error_series(run_record: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    timesteps, errors = _metric_series(run_record["metrics_rows"], ROLLOUT_ERROR_KEYS)
    if len(timesteps):
        return timesteps, errors
    return None, None


def _latest_value_at_or_before(
    timesteps: np.ndarray,
    values: np.ndarray,
    checkpoint: int,
) -> float | None:
    if len(timesteps) == 0:
        return None
    eligible = np.where(timesteps <= checkpoint)[0]
    if len(eligible) == 0:
        return None
    return float(values[int(eligible[-1])])


def _nice_step(value: int) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** max(len(str(value)) - 1, 0)
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= value:
            return candidate
    return 10 * magnitude


def _shared_checkpoints(run_records: list[tuple[dict, np.ndarray, np.ndarray]], n_points: int) -> tuple[int, ...]:
    if not run_records:
        return tuple()
    shared_max = min(int(timesteps.max()) for _, timesteps, _ in run_records if len(timesteps) > 0)
    if shared_max <= 0:
        return tuple()
    raw_points = np.linspace(shared_max / max(n_points, 1), shared_max, num=n_points)
    checkpoints = []
    for point in raw_points:
        checkpoint = _nice_step(int(point))
        checkpoint = min(checkpoint, shared_max)
        checkpoints.append(checkpoint)
    ordered = []
    for checkpoint in checkpoints:
        if checkpoint not in ordered:
            ordered.append(checkpoint)
    return tuple(ordered)


def plot_reward_curves(run_records: dict[str, dict], save_path="logs/fig1_reward_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ["#95a5a6", "#e67e22", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for i, record in enumerate(run_records.values()):
        timesteps, rewards = load_reward_series(record)
        if timesteps is None:
            continue
        ax.plot(timesteps, rewards, label=record["algorithm"], color=colors[i % len(colors)], linewidth=2)
        plotted += 1

    if plotted == 0:
        raise ValueError("No reward series found in artifacts/logs.")

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Training Performance From Saved Artifacts", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_planning_stability(results: dict, save_path="logs/fig3_planning_stability.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [("MLP", "#e74c3c"), ("S5", "#2ecc71")]
    x = np.array([5, 10])
    bar_width = 0.35
    for i, (name, color) in enumerate(groups):
        heights = [results.get(f"{name} H=5", 0), results.get(f"{name} H=10", 0)]
        ax.bar(x + i * bar_width, heights, bar_width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(["Horizon 5", "Horizon 10"], fontsize=12)
    ax.set_ylabel("Final Mean Reward", fontsize=13)
    ax.set_title("Planning Stability: MLP vs SSM at Different Horizons", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_sample_efficiency(run_records: dict[str, dict], checkpoints=(50_000, 100_000, 200_000), save_path="logs/fig4_sample_efficiency.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    valid_records = []
    for record in run_records.values():
        timesteps, rewards = load_reward_series(record)
        if timesteps is not None and len(timesteps) > 0:
            valid_records.append((record, timesteps, rewards))

    resolved_checkpoints = tuple(checkpoints)
    supported_runs = 0
    for checkpoint in resolved_checkpoints:
        supported_runs += sum(
            _latest_value_at_or_before(timesteps, rewards, checkpoint) is not None
            for _, timesteps, rewards in valid_records
        )
    if supported_runs == 0:
        resolved_checkpoints = _shared_checkpoints(valid_records, len(checkpoints))
    if not resolved_checkpoints:
        raise ValueError("No checkpoints could be derived from available run data.")

    x = np.arange(len(resolved_checkpoints))
    bar_width = 0.8 / max(len(valid_records), 1)

    for i, (record, timesteps, rewards) in enumerate(valid_records):
        heights = []
        for checkpoint in resolved_checkpoints:
            value = _latest_value_at_or_before(timesteps, rewards, checkpoint)
            heights.append(np.nan if value is None else value)
        ax.bar(x + i * bar_width, heights, bar_width, label=record["algorithm"], alpha=0.85)

    ax.set_xticks(x + bar_width * max(len(valid_records) - 1, 0) / 2)
    ax.set_xticklabels([f"{ck // 1000}k steps" for ck in resolved_checkpoints], fontsize=11)
    ax.set_ylabel("Mean Reward at Checkpoint", fontsize=13)
    ax.set_title("Sample Efficiency Comparison", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
