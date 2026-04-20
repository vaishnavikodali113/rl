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
import { useArtifact } from "../hooks/use-artifact";
import { ALGO_COLORS, DEFAULT_COLOR, EnvironmentFilter, inferEnvironment } from "../lib/constants";

interface TrainingData {
  [algo: string]: { timesteps: number[]; rewards: number[] };
}

interface Props {
  envFilter: EnvironmentFilter;
}

export function TrainingCurves({ envFilter }: Props) {
  const { data, loading, error } = useArtifact<TrainingData>(
    "/artifacts/reward-curves"
  );

  if (loading)
    return (
      <div className="h-96 flex items-center justify-center text-zinc-500 dark:text-zinc-500">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
          <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          <span className="ml-2 font-medium">Loading training curves...</span>
        </div>
      </div>
    );
  if (error) return <p className="text-red-400 text-sm">Error: {error}</p>;
  if (!data) return null;

  const filteredEntries = Object.entries(data).filter(([algo]) => {
    return envFilter === "all" || inferEnvironment(algo) === envFilter;
  });

  if (filteredEntries.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-zinc-500 dark:text-zinc-500">
        No training curves for the selected environment yet.
      </div>
    );
  }

  const filteredData = Object.fromEntries(filteredEntries);

  const allTimesteps = Array.from(
    new Set(Object.values(filteredData).flatMap((d) => d.timesteps))
  ).sort((a, b) => a - b);

  const chartData = allTimesteps.map((t) => {
    const point: Record<string, number | undefined> = { timestep: t };
    Object.entries(filteredData).forEach(([algo, d]) => {
      const idx = d.timesteps.indexOf(t);
      point[algo] = idx >= 0 ? d.rewards[idx] : undefined;
    });
    return point;
  });

  const algos = Object.keys(filteredData);

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" className="dark:opacity-20" />
          <XAxis
            dataKey="timestep"
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
            tick={{ fontSize: 12, fill: "#71717a" }}
            stroke="#71717a"
            label={{
              value: "Environment Steps",
              position: "insideBottomRight",
              offset: -5,
              style: { fill: "#71717a", fontSize: 11 },
            }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#71717a" }}
            stroke="#71717a"
            label={{
              value: "Mean Episode Reward",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 11 },
            }}
          />
          <Tooltip
            formatter={(v: number) => v.toFixed(1)}
            labelFormatter={(v: number) => `${(v / 1000).toFixed(0)}k steps`}
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
          {algos.map((algo) => (
            <Line
              key={algo}
              type="monotone"
              dataKey={algo}
              stroke={ALGO_COLORS[algo] ?? DEFAULT_COLOR}
              dot={false}
              strokeWidth={3}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
