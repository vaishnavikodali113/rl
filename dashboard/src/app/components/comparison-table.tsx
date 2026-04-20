import { TrendingUp, Zap, Target, Clock } from "lucide-react";
import { useArtifact } from "../hooks/use-artifact";

interface Row {
  algorithm: string;
  mean_reward: string;
  std_reward: string;
  [key: string]: string;
}

const COLUMN_ICONS: Record<string, any> = {
  mean_reward: TrendingUp,
  final_reward: TrendingUp,
  sample_efficiency: Zap,
  rollout_error: Target,
  wall_time: Clock,
};

export function ComparisonTable() {
  const { data, loading, error } = useArtifact<Row[]>(
    "/artifacts/comparison-table"
  );

  if (loading)
    return (
      <div className="h-64 flex items-center justify-center text-zinc-500 dark:text-zinc-500">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
          <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          <span className="ml-2 font-medium">Loading comparison data...</span>
        </div>
      </div>
    );
  if (error) return <p className="text-red-400 text-sm">Error: {error}</p>;
  if (!data || data.length === 0)
    return (
      <p className="text-zinc-500 dark:text-zinc-500 text-sm">
        No data yet — run Phase 4 evaluation.
      </p>
    );

  const columns = Object.keys(data[0]);

  return (
    <div className="overflow-x-auto rounded-2xl border border-zinc-200 dark:border-zinc-800 shadow-xl dark:shadow-none">
      <table className="min-w-full text-sm">
        <thead className="bg-zinc-100/80 dark:bg-zinc-900/50 border-b border-zinc-200 dark:border-zinc-800">
          <tr>
            {columns.map((col) => {
              const Icon = COLUMN_ICONS[col];
              return (
                <th
                  key={col}
                  className="px-5 py-4 text-left font-bold text-zinc-700 dark:text-zinc-300 capitalize"
                >
                  <div className="flex items-center gap-2">
                    {Icon && <Icon className="w-4 h-4 text-zinc-500 dark:text-zinc-400" />}
                    {col.replace(/_/g, " ")}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-200/50 dark:divide-zinc-800/50 bg-white/60 dark:bg-zinc-900/30">
          {data.map((row, i) => (
            <tr
              key={i}
              className="hover:bg-blue-50/50 dark:hover:bg-white/5 transition-all duration-150"
            >
              {columns.map((col, j) => (
                <td
                  key={col}
                  className={`px-5 py-4 ${
                    j === 0
                      ? "font-semibold text-zinc-800 dark:text-zinc-100"
                      : "font-mono text-zinc-700 dark:text-zinc-200 font-semibold"
                  } text-xs`}
                >
                  {row[col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
