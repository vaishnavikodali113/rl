from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

from evaluation.benchmark import benchmark_update_step
from evaluation.compare_plots import (
    discover_runs,
    load_reward_series,
    load_rollout_error_series,
    plot_reward_curves,
    plot_sample_efficiency,
)


def _device_for_eval() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_path(summary: dict) -> str | None:
    artifacts = summary.get("artifacts", {})
    candidates = [
        Path("logs") / summary["run_name"] / "best" / "best_model.pt",
        Path(str(artifacts.get("model", ""))),
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _build_model_from_checkpoint(run_record: dict, device: str):
    try:
        from tdmpc2.model import TDMPC2Model
    except ModuleNotFoundError as exc:
        print(f"Skipping checkpoint model load for {run_record['run_name']}: {exc}")
        return None

    summary = run_record["summary"]
    algorithm = str(summary.get("algorithm", "")).lower()
    if "td-mpc2" not in algorithm:
        return None

    checkpoint_path = _checkpoint_path(summary)
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    latent_dim = int(config.get("latent_dim", model_state["reward.net.0.weight"].shape[1] - 6))
    obs_dim = int(model_state["encoder.net.0.weight"].shape[1])
    action_dim = int(model_state["reward.net.0.weight"].shape[1] - latent_dim)
    dynamics_type = summary["run_name"].split("_")[-1]
    if dynamics_type == "h10":
        dynamics_type = "s5"

    model = TDMPC2Model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        dynamics_type=dynamics_type,
        ssm_state_dim=int(config.get("ssm_state_dim", 256)),
        simnorm_dim=int(config.get("simnorm_dim", 8)) if config.get("simnorm_dim", 8) > 0 else None,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print(
            f"Checkpoint compatibility note for {summary['run_name']}: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    model.planner_config = {
        "plan_horizon": config.get("plan_horizon", 5),
        "plan_samples": config.get("plan_samples", 512),
        "plan_temperature": config.get("plan_temperature", 0.5),
        "gamma": config.get("gamma", 0.99),
    }
    model.to(device)
    model.eval()
    return model


def _format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}"


def _last_value(series: tuple | None) -> float | None:
    if not series:
        return None
    _, values = series
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def _build_comparison_rows(run_records: dict[str, dict], device: str) -> list[dict[str, str]]:
    rows = []
    for run_name, record in run_records.items():
        summary = record["summary"]
        reward_series = load_reward_series(record)
        error_series = load_rollout_error_series(record)
        final_reward = _last_value(reward_series)
        latent_mse = _last_value(error_series)
        benchmark_ms = None

        model = _build_model_from_checkpoint(record, device=device)
        if model is not None:
            try:
                benchmark_ms = benchmark_update_step(
                    model,
                    batch_size=int(summary.get("config", {}).get("batch_size", 128)),
                    horizon=int(summary.get("config", {}).get("plan_horizon", 5)),
                    n_runs=25,
                    device=device,
                )
            except Exception as exc:
                print(f"Benchmark failed for {run_name}: {exc}")

        rows.append(
            {
                "run_name": run_name,
                "algorithm": summary.get("algorithm", run_name),
                "environment": summary.get("environment", "---"),
                "total_timesteps": str(summary.get("total_timesteps", "---")),
                "best_eval_mean_reward": _format_metric(summary.get("best_eval_mean_reward"), digits=3),
                "final_logged_reward": _format_metric(final_reward, digits=3),
                "final_latent_mse": _format_metric(latent_mse, digits=6),
                "ms_per_world_model_rollout": _format_metric(benchmark_ms, digits=2) if benchmark_ms is not None else "---",
                "metrics_jsonl": str(record.get("metrics_path") or "---"),
                "eval_npz": str(record.get("eval_npz_path") or "---"),
            }
        )
    return rows


def _write_comparison_table(rows: list[dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "run_name",
        "algorithm",
        "environment",
        "total_timesteps",
        "best_eval_mean_reward",
        "final_logged_reward",
        "final_latent_mse",
        "ms_per_world_model_rollout",
        "metrics_jsonl",
        "eval_npz",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(run_records: dict[str, dict], rows: list[dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tdmpc_rows = [row for row in rows if "TD-MPC2" in row["algorithm"] and row["final_latent_mse"] != "---"]
    tdmpc_rows.sort(key=lambda row: float(row["final_latent_mse"]), reverse=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Evaluation Report\n\n")
        handle.write("This report is sourced from saved `artifacts/*/metrics.jsonl`, `artifacts/*/summary.json`, and `logs/*/eval/evaluations.npz`.\n\n")
        if tdmpc_rows:
            handle.write("## Latent MSE Ranking\n\n")
            for row in tdmpc_rows:
                handle.write(
                    f"- {row['run_name']}: latent MSE `{row['final_latent_mse']}`, "
                    f"final logged reward `{row['final_logged_reward']}`\n"
                )
            handle.write("\n")


def main(artifacts_root: str = "artifacts", output_dir: str = "artifacts/evaluation_pngs") -> None:
    print("Running evaluation from saved artifacts and logs...")
    os.makedirs(output_dir, exist_ok=True)
    device = _device_for_eval()

    run_records = discover_runs(artifacts_root=artifacts_root)
    if not run_records:
        raise FileNotFoundError(f"No run summaries found under {artifacts_root!r}.")

    print("Plotting reward curves...")
    plot_reward_curves(run_records, os.path.join(output_dir, "fig1_reward_curves.png"))

    print("Plotting sample efficiency...")
    plot_sample_efficiency(run_records, save_path=os.path.join(output_dir, "fig4_sample_efficiency.png"))

    rows = _build_comparison_rows(run_records, device=device)
    comparison_table_path = os.path.join(output_dir, "comparison_table.csv")
    _write_comparison_table(rows, comparison_table_path)
    _write_report(run_records, rows, os.path.join(output_dir, "report.md"))

    print(f"Wrote {comparison_table_path}")
    print(f"Evaluation execution completed. Check {output_dir}/")


if __name__ == "__main__":
    main()
