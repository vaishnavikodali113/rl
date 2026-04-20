import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@radix-ui/react-tabs";
import { useWebSocket } from "./hooks/use-websocket";
import { useArtifact } from "./hooks/use-artifact";
import { VideoPanel } from "./components/video-panel";
import { LiveRewardChart } from "./components/live-reward-chart";
import { TrainingCurves } from "./components/training-curves";
import { RolloutErrorChart } from "./components/rollout-error-chart";
import { ComparisonTable } from "./components/comparison-table";
import { ThemeProvider } from "./components/theme-provider";
import { ThemeToggle } from "./components/theme-toggle";
import { Sparkles } from "lucide-react";
import { EnvironmentFilter } from "./lib/constants";

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

function AppContent() {
  const { message, connected } = useWebSocket();
  const { data: health } = useArtifact<{
    status: string;
    models: Array<{
      label: string;
      run_name: string;
      display_name: string;
      algorithm_name: string;
      env_name: string;
      task: string;
      env_theme: string;
      env_title: string;
      behavior_text: string;
    }>;
    startup_error?: string | null;
    stream_error?: string | null;
  }>("/health");
  const [activeTab, setActiveTab] = useState("live");
  const [envFilter, setEnvFilter] = useState<EnvironmentFilter>("all");
  const liveModels = health?.models ?? [];
  const hasLiveModels = liveModels.length > 0;
  const liveCards = message?.models ?? liveModels;
  const filteredLiveEntries = liveCards
    .map((model, index) => ({
      model,
      frame: message?.frames[index] ?? "",
      metric: message?.metrics.find((item) => item.label === model.label),
    }))
    .filter(({ model }) => envFilter === "all" || model.env_name === envFilter);
  const liveError = health?.stream_error ?? health?.startup_error ?? null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-zinc-100 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors duration-300">
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md sticky top-0 z-10 shadow-sm dark:shadow-none">
        <div className="px-6 py-4 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
                Efficient MBRL with SSM World Models
              </h1>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-500 mt-1 ml-10">
              TD-MPC2 · S4 · S5 · Mamba — Riddhi Poddar · Vedant Shinde · Soham
              Kulkarni · Sri Vaishnavi Kodali
            </p>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/50 backdrop-blur-sm">
          <TabsList className="flex gap-2 px-6">
            <TabsTrigger
              value="live"
              className={`px-5 py-3 text-sm font-semibold transition-all duration-200 rounded-t-xl relative ${
                activeTab === "live"
                  ? "bg-gradient-to-br from-blue-500/10 to-purple-500/10 dark:from-blue-500/20 dark:to-purple-500/20 text-blue-700 dark:text-blue-300 border-t-2 border-x border-blue-500 dark:border-blue-400 border-b-0 shadow-lg"
                  : "text-zinc-600 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
              }`}
            >
              Live Rollout
            </TabsTrigger>
            <TabsTrigger
              value="training"
              className={`px-5 py-3 text-sm font-semibold transition-all duration-200 rounded-t-xl ${
                activeTab === "training"
                  ? "bg-gradient-to-br from-emerald-500/10 to-teal-500/10 dark:from-emerald-500/20 dark:to-teal-500/20 text-emerald-700 dark:text-emerald-300 border-t-2 border-x border-emerald-500 dark:border-emerald-400 border-b-0 shadow-lg"
                  : "text-zinc-600 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
              }`}
            >
              Training Curves
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className={`px-5 py-3 text-sm font-semibold transition-all duration-200 rounded-t-xl ${
                activeTab === "analysis"
                  ? "bg-gradient-to-br from-pink-500/10 to-rose-500/10 dark:from-pink-500/20 dark:to-rose-500/20 text-pink-700 dark:text-pink-300 border-t-2 border-x border-pink-500 dark:border-pink-400 border-b-0 shadow-lg"
                  : "text-zinc-600 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
              }`}
            >
              Rollout Error
            </TabsTrigger>
            <TabsTrigger
              value="results"
              className={`px-5 py-3 text-sm font-semibold transition-all duration-200 rounded-t-xl ${
                activeTab === "results"
                  ? "bg-gradient-to-br from-amber-500/10 to-orange-500/10 dark:from-amber-500/20 dark:to-orange-500/20 text-amber-700 dark:text-amber-300 border-t-2 border-x border-amber-500 dark:border-amber-400 border-b-0 shadow-lg"
                  : "text-zinc-600 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
              }`}
            >
              Results
            </TabsTrigger>
          </TabsList>
        </div>

        <main className="px-6 py-6">
          <div className="mb-6 flex flex-wrap items-center gap-3 rounded-2xl border border-zinc-200 bg-white/70 p-3 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/40">
            <span className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500 dark:text-zinc-400">
              Environment Dial
            </span>
            {(["all", "walker", "cheetah"] as EnvironmentFilter[]).map((env) => (
              <button
                key={env}
                type="button"
                onClick={() => setEnvFilter(env)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  envFilter === env
                    ? "bg-zinc-900 text-white dark:bg-white dark:text-zinc-950"
                    : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
                }`}
              >
                {env === "all" ? "All Runs" : env[0].toUpperCase() + env.slice(1)}
              </button>
            ))}
          </div>

          <TabsContent value="live" className="space-y-6">
            <div className="flex items-center gap-3 text-sm">
              <div className={`relative w-3 h-3 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`}>
                {connected && (
                  <span className="absolute inset-0 rounded-full bg-emerald-500 animate-ping opacity-75" />
                )}
              </div>
              <span className="text-zinc-700 dark:text-zinc-400 font-mono text-xs font-medium">
                {connected ? "Connected to backend" : "Reconnecting…"}
              </span>
            </div>

            {message ? (
              <>
                {filteredLiveEntries.length > 0 ? (
                  <div
                    className="grid gap-4"
                    style={{
                      gridTemplateColumns: `repeat(${Math.min(filteredLiveEntries.length, 3)}, 1fr)`,
                    }}
                  >
                    {filteredLiveEntries.map(({ model, frame, metric }) => (
                      <VideoPanel
                        key={model.label}
                        model={model}
                        frame={frame}
                        metric={metric}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900/50 dark:text-zinc-400">
                    No live runs match the current environment dial.
                  </div>
                )}

                <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/50 backdrop-blur-sm p-6 shadow-xl dark:shadow-none">
                  <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-5">
                    Live Step Reward
                  </h2>
                  <LiveRewardChart
                    labels={message.labels}
                    latestMetrics={message.metrics}
                    envFilter={envFilter}
                  />
                </div>
              </>
            ) : (
              <div
                className={`text-sm rounded-lg p-4 border ${
                  liveError
                    ? "text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900/50"
                    : "text-zinc-500 dark:text-zinc-500 bg-zinc-100 dark:bg-zinc-900/50 border-zinc-200 dark:border-zinc-800"
                }`}
              >
                {liveError
                  ? `Live rollout is blocked: ${liveError}`
                  : connected
                    ? hasLiveModels
                      ? "Waiting for the first live environment frame…"
                      : "Backend is up, but no live checkpoints were found. The loader now searches both artifacts/* and logs/<run>/best or final_model* automatically."
                    : "Backend offline — start the FastAPI server to stream the live environments."}
              </div>
            )}
          </TabsContent>

          <TabsContent value="training" className="space-y-5">
            <div>
              <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                Fig 1 — Training Reward Curves
              </h2>
              <p className="text-sm text-zinc-600 dark:text-zinc-500 mt-1">
                All algorithms — reward vs. environment steps. Loaded from{" "}
                <code className="text-zinc-700 dark:text-zinc-400 font-mono text-xs bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  artifacts/*/metrics.jsonl
                </code>
                .
              </p>
            </div>
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/50 backdrop-blur-sm p-6 shadow-xl dark:shadow-none">
              <TrainingCurves envFilter={envFilter} />
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-8">
            <div className="space-y-3">
              <div>
                <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  Fig 2 — Multi-Step Rollout Error
                </h2>
                <p className="text-sm text-zinc-600 dark:text-zinc-500 mt-1">
                  Latent MSE at each prediction horizon. Loaded from{" "}
                  <code className="text-zinc-700 dark:text-zinc-400 font-mono text-xs bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                    artifacts/*/rollout_errors.npy
                  </code>
                  .
                </p>
              </div>
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/50 backdrop-blur-sm p-6 shadow-xl dark:shadow-none">
                <RolloutErrorChart envFilter={envFilter} />
              </div>
            </div>

            <div className="rounded-xl border border-blue-200 dark:border-blue-900/50 bg-blue-50/50 dark:bg-blue-950/20 p-5 text-sm text-zinc-700 dark:text-zinc-400">
              <strong className="text-blue-700 dark:text-blue-300">Fig 4 — Sample Efficiency</strong>{" "}
              is derived from the training curves on the Training Curves tab —
              read reward values at 50k, 100k, and 200k steps from the chart
              above. A dedicated bar chart view can be added by extending the
              training curves component with checkpoint markers.
            </div>
          </TabsContent>

          <TabsContent value="results" className="space-y-5">
            <div>
              <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                Final Comparison Table
              </h2>
              <p className="text-sm text-zinc-600 dark:text-zinc-500 mt-1">
                Loaded from{" "}
                <code className="text-zinc-700 dark:text-zinc-400 font-mono text-xs bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  artifacts/comparison_table.csv
                </code>
                . Generated by{" "}
                <code className="text-zinc-700 dark:text-zinc-400 font-mono text-xs bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  evaluation/eval_runner.py
                </code>
                .
              </p>
            </div>
            <ComparisonTable envFilter={envFilter} />
          </TabsContent>
        </main>
      </Tabs>
    </div>
  );
}
