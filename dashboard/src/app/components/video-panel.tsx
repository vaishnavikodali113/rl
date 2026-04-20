import { Activity, Zap } from "lucide-react";
import { LiveModelCard, StepMetric } from "../hooks/use-websocket";
import { ALGO_COLORS, DEFAULT_COLOR } from "../lib/constants";

interface Props {
  model: LiveModelCard;
  frame: string;
  metric: StepMetric | undefined;
}

export function VideoPanel({ model, frame, metric }: Props) {
  const color = ALGO_COLORS[model.label] ?? DEFAULT_COLOR;
  const envBadgeClass =
    model.env_name === "cheetah"
      ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300"
      : "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-300";

  return (
    <div className="flex flex-col rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] group">
      <div
        className="flex items-center gap-3 px-4 py-3 border-b-2 bg-gradient-to-r from-transparent via-transparent to-transparent hover:from-white/50 dark:hover:from-zinc-900/50 transition-colors"
        style={{ borderBottomColor: color }}
      >
        <span
          className="inline-block w-3 h-3 rounded-full shadow-lg animate-pulse"
          style={{ background: color, boxShadow: `0 0 10px ${color}` }}
        />
        <div className="min-w-0">
          <div className="text-sm font-bold text-zinc-900 dark:text-zinc-100 truncate">
            {model.run_name}
          </div>
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400 truncate">
            <span>{model.algorithm_name}</span>
            <span className={`rounded-full border px-2 py-0.5 tracking-[0.14em] ${envBadgeClass}`}>
              {model.env_name}
            </span>
          </div>
        </div>
        {metric && (
          <div className="ml-auto flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1 text-zinc-600 dark:text-zinc-400">
              <Activity className="w-3.5 h-3.5" style={{ color }} />
              <span className="font-mono font-semibold">{metric.reward.toFixed(2)}</span>
            </span>
            <span className="text-zinc-500 dark:text-zinc-500">|</span>
            <span className="font-mono text-zinc-700 dark:text-zinc-300 font-semibold">
              ep: {metric.episode_reward.toFixed(0)}
            </span>
          </div>
        )}
      </div>

      {frame ? (
        <div className="relative">
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt={model.run_name}
            className="w-full object-cover"
            style={{ aspectRatio: "4/3" }}
          />
          <div className="absolute left-3 top-3 rounded-full border border-white/25 bg-black/45 px-3 py-1 text-[11px] font-semibold tracking-[0.16em] text-white backdrop-blur-sm">
            {model.behavior_text}
          </div>
          {metric && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="flex items-center gap-4 text-white text-xs">
                <div className="flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5" />
                  <span className="font-mono">{metric.action_magnitude.toFixed(3)}</span>
                </div>
                {(metric.low_motion_steps ?? 0) >= 8 && (
                  <span className="rounded-full bg-white/15 px-2 py-0.5">
                    rescue {metric.low_motion_steps}
                  </span>
                )}
                <span>Step {metric.step}</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div
          className="w-full bg-zinc-100 dark:bg-zinc-900 flex items-center justify-center text-zinc-400 dark:text-zinc-600 text-sm"
          style={{ aspectRatio: "4/3" }}
        >
          <div className="flex flex-col items-center gap-2">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span>Waiting for frames…</span>
          </div>
        </div>
      )}
    </div>
  );
}
