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
from tdmpc2.compat import load_tdmpc2_agent


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
    summary = run_record["summary"]
    algorithm = str(summary.get("algorithm", "")).lower()
    if "td-mpc2" not in algorithm:
        return None

    checkpoint_path = _checkpoint_path(summary)
    if checkpoint_path is None:
        return None

    summary_path = Path(str(summary.get("artifacts", {}).get("summary_json", "")))
    if not summary_path.is_file():
        summary_path = Path("artifacts") / summary["run_name"] / "summary.json"
    try:
        return load_tdmpc2_agent(checkpoint_path, summary_path=summary_path, device=device)
    except Exception as exc:
        print(f"Failed to reconstruct TD-MPC2 agent for {summary['run_name']}: {exc}")
        return None


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
    print(f"Phase 4 execution completed. Check {output_dir}/")


if __name__ == "__main__":
    main()
