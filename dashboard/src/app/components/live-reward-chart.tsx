import { useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { StepMetric } from "../hooks/use-websocket";
import { ALGO_COLORS, DEFAULT_COLOR, EnvironmentFilter } from "../lib/constants";

interface Props {
  labels: string[];
  latestMetrics: StepMetric[];
  envFilter: EnvironmentFilter;
}

const HISTORY_LIMIT = 300;
const historyRef: { current: Record<string, { step: number; reward: number }[]> } = {
  current: {},
};

export function LiveRewardChart({ labels, latestMetrics, envFilter }: Props) {
  latestMetrics.forEach((m) => {
    if (!historyRef.current[m.label]) historyRef.current[m.label] = [];
    const arr = historyRef.current[m.label];
    arr.push({ step: m.step, reward: m.reward });
    if (arr.length > HISTORY_LIMIT) arr.shift();
  });

  const filteredLabels = labels.filter((label) => {
    if (envFilter === "all") return true;
    const metric = latestMetrics.find((item) => item.label === label);
    return metric?.env_name === envFilter;
  });

  const maxLen = Math.max(
    0,
    ...filteredLabels.map((l) => historyRef.current[l]?.length ?? 0)
  );
  const chartData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = {};
    filteredLabels.forEach((l) => {
      const arr = historyRef.current[l];
      if (arr && arr[i] !== undefined) {
        point["step"] = arr[i].step;
        point[l] = arr[i].reward;
      }
    });
    return point;
  });

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" className="dark:opacity-20" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 12, fill: "#71717a" }}
            stroke="#71717a"
            label={{
              value: "Recent Steps",
              position: "insideBottom",
              offset: -5,
              style: { fill: "#71717a", fontSize: 11 },
            }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#71717a" }}
            stroke="#71717a"
            label={{
              value: "Reward",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 11 },
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.98)",
              border: "1px solid rgba(0,0,0,0.1)",
              borderRadius: "12px",
              boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)",
              color: "#18181b",
            }}
            labelStyle={{ color: "#18181b", fontWeight: 600 }}
            wrapperClassName="dark:[&>div]:!bg-zinc-900/98 dark:[&>div]:!border-white/10 dark:[&>div]:!text-zinc-100"
          />
          <Legend wrapperStyle={{ fontSize: "12px", fontWeight: 500 }} />
          {filteredLabels.map((l) => (
            <Line
              key={l}
              type="monotone"
              dataKey={l}
              stroke={ALGO_COLORS[l] ?? DEFAULT_COLOR}
              dot={false}
              strokeWidth={3}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
