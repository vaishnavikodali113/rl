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

        if result:
            return result

        # Fall back to the aggregated evaluation CSV used by the report assets.
        csv_path = os.path.join(ARTIFACTS_DIR, "evaluation_pngs", "rollout_error_stats.csv")
        if not os.path.isfile(csv_path):
            return {}

        import csv

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row["algorithm"].strip()
                means = []
                for horizon in range(1, 11):
                    value = row.get(f"h{horizon}_mean")
                    if value in (None, ""):
                        continue
                    means.append(float(value))
                if means:
                    result[label] = means

        return result

    @staticmethod
    def load_comparison_table() -> list[dict]:
        """
        Reads artifacts/evaluation_pngs/comparison_table.csv.
        Used by the /artifacts/comparison-table REST endpoint.
        """
        import csv
        path = os.path.join(ARTIFACTS_DIR, "evaluation_pngs", "comparison_table.csv")
        if not os.path.isfile(path):
            return []
            
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                
        return rows
