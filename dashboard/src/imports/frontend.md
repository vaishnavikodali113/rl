# RL Dashboard Frontend — `frontend.md`

> **Stack:** Next.js 14 (App Router) · TypeScript · Recharts · Tailwind CSS · shadcn/ui
> **Connects to:** FastAPI backend at `http://localhost:8000` (see `server.md`)
> **Reads:** WebSocket stream `/ws` + REST endpoints for static artifact charts
> **What it shows:** Live side-by-side model rollout videos · Real-time reward graph · Training reward curves (Fig 1) · Rollout error chart (Fig 2) · Sample efficiency chart (Fig 4) · Comparison table

---

## Where This Lives

```
rl/
├── server/         ← backend (see server.md)
├── dashboard/      ← THIS — the Next.js project lives here
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx              ← root: redirects to /live
│   │   ├── live/
│   │   │   └── page.tsx          ← Live rollout tab
│   │   ├── training/
│   │   │   └── page.tsx          ← Training curves tab (Fig 1)
│   │   ├── analysis/
│   │   │   └── page.tsx          ← Rollout error + sample efficiency (Fig 2, 4)
│   │   └── results/
│   │       └── page.tsx          ← Comparison table
│   ├── components/
│   │   ├── nav.tsx               ← Tab navigation
│   │   ├── video-panel.tsx       ← Single model video display
│   │   ├── live-reward-chart.tsx ← Rolling reward line chart (live)
│   │   ├── training-curves.tsx   ← Static training reward curves
│   │   ├── rollout-error-chart.tsx
│   │   ├── sample-efficiency-chart.tsx
│   │   └── comparison-table.tsx
│   ├── hooks/
│   │   ├── use-websocket.ts      ← WebSocket connection + message parser
│   │   └── use-artifact.ts       ← Generic REST fetch hook
│   └── lib/
│       └── constants.ts          ← Colors per algorithm, API base URL
```

---

## Step 1 — Scaffold the Project

Run from inside `rl/dashboard/` (create the directory first):

```bash
mkdir -p rl/dashboard && cd rl/dashboard

npx create-next-app@latest . \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --no-src-dir \
  --import-alias "@/*"
```

Then install charting and UI libraries:

```bash
npm install recharts
npm install @radix-ui/react-tabs @radix-ui/react-badge
npm install lucide-react
npx shadcn@latest init      # follow prompts: use default config
npx shadcn@latest add table badge tabs card
```

---

## Step 2 — `lib/constants.ts`

Centralised color map matching Phase 4 `compare_plots.py` colors, and the API base URL.

```typescript
// dashboard/lib/constants.ts

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const WS_URL   = process.env.NEXT_PUBLIC_WS_URL  ?? "ws://localhost:8000/ws";

/** Color per algorithm — matches compare_plots.py palette */
export const ALGO_COLORS: Record<string, string> = {
  "ppo_walker":           "#95a5a6",
  "sac_cheetah":          "#e67e22",
  "tdmpc2_walker_mlp":    "#e74c3c",
  "tdmpc2_walker_s4":     "#3498db",
  "tdmpc2_walker_s5":     "#2ecc71",
  "tdmpc2_walker_mamba":  "#9b59b6",
  // Live stream labels (from server registry)
  "PPO (Walker)":         "#95a5a6",
  "SAC (Cheetah)":        "#e67e22",
  "TD-MPC2 MLP":          "#e74c3c",
  "TD-MPC2 S4":           "#3498db",
  "TD-MPC2 S5":           "#2ecc71",
  "TD-MPC2 Mamba":        "#9b59b6",
};

export const DEFAULT_COLOR = "#8884d8";
```

---

## Step 3 — `hooks/use-websocket.ts`

Manages the WebSocket lifecycle and parses incoming frames + metrics.

```typescript
// dashboard/hooks/use-websocket.ts
"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { WS_URL } from "@/lib/constants";

export interface StepMetric {
  label: string;
  step: number;
  reward: number;
  episode_reward: number;
  done: boolean;
  action_magnitude: number;
}

export interface WSMessage {
  labels: string[];
  frames: string[];          // base64 JPEG strings
  metrics: StepMetric[];
}

export function useWebSocket() {
  const [message, setMessage] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen  = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      try {
        setMessage(JSON.parse(event.data) as WSMessage);
      } catch {
        // ignore malformed frames
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { message, connected };
}
```

---

## Step 4 — `hooks/use-artifact.ts`

Generic hook that fetches any REST endpoint once and returns the data.

```typescript
// dashboard/hooks/use-artifact.ts
"use client";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/constants";

export function useArtifact<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}${path}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [path]);

  return { data, loading, error };
}
```

---

## Step 5 — `components/video-panel.tsx`

Displays one model's live video frame with its label and latest reward overlay.

```tsx
// dashboard/components/video-panel.tsx
"use client";
import { StepMetric } from "@/hooks/use-websocket";
import { ALGO_COLORS, DEFAULT_COLOR } from "@/lib/constants";

interface Props {
  label: string;
  frame: string;          // base64 JPEG
  metric: StepMetric | undefined;
}

export function VideoPanel({ label, frame, metric }: Props) {
  const color = ALGO_COLORS[label] ?? DEFAULT_COLOR;

  return (
    <div className="flex flex-col rounded-xl overflow-hidden border border-zinc-700 bg-zinc-900">
      {/* Label bar */}
      <div className="flex items-center gap-2 px-3 py-2" style={{ borderBottom: `2px solid ${color}` }}>
        <span className="inline-block w-3 h-3 rounded-full" style={{ background: color }} />
        <span className="text-sm font-semibold text-zinc-100 truncate">{label}</span>
        {metric && (
          <span className="ml-auto text-xs text-zinc-400">
            r={metric.reward.toFixed(2)} &nbsp;|&nbsp; ep={metric.episode_reward.toFixed(0)}
          </span>
        )}
      </div>

      {/* Video frame */}
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt={label}
          className="w-full object-cover"
          style={{ aspectRatio: "4/3" }}
        />
      ) : (
        <div className="w-full bg-zinc-800 flex items-center justify-center text-zinc-500 text-sm"
             style={{ aspectRatio: "4/3" }}>
          Waiting for frames…
        </div>
      )}
    </div>
  );
}
```

---

## Step 6 — `components/live-reward-chart.tsx`

Rolling reward line chart updated on every WebSocket frame. Keeps the last 300 steps per model.

```tsx
// dashboard/components/live-reward-chart.tsx
"use client";
import { useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { StepMetric } from "@/hooks/use-websocket";
import { ALGO_COLORS, DEFAULT_COLOR } from "@/lib/constants";

interface Props {
  labels: string[];
  latestMetrics: StepMetric[];
}

// Kept outside component to avoid re-render resets
const HISTORY_LIMIT = 300;
const historyRef: { current: Record<string, { step: number; reward: number }[]> } = { current: {} };

export function LiveRewardChart({ labels, latestMetrics }: Props) {
  // Append latest data point per label
  latestMetrics.forEach((m) => {
    if (!historyRef.current[m.label]) historyRef.current[m.label] = [];
    const arr = historyRef.current[m.label];
    arr.push({ step: m.step, reward: m.reward });
    if (arr.length > HISTORY_LIMIT) arr.shift();
  });

  // Build a unified time-series array for Recharts
  const maxLen = Math.max(...labels.map((l) => (historyRef.current[l]?.length ?? 0)));
  const chartData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = {};
    labels.forEach((l) => {
      const arr = historyRef.current[l];
      if (arr && arr[i] !== undefined) {
        point["step"] = arr[i].step;
        point[l] = arr[i].reward;
      }
    });
    return point;
  });

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <XAxis dataKey="step" tick={{ fontSize: 11 }} label={{ value: "Step", position: "insideBottomRight", offset: -5 }} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "Reward", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          {labels.map((l) => (
            <Line
              key={l}
              type="monotone"
              dataKey={l}
              stroke={ALGO_COLORS[l] ?? DEFAULT_COLOR}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

## Step 7 — `components/training-curves.tsx`

Static chart that loads the full training history from `artifacts/*/metrics.jsonl` via the REST endpoint. This is Fig 1 from Phase 4.

```tsx
// dashboard/components/training-curves.tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { useArtifact } from "@/hooks/use-artifact";
import { ALGO_COLORS, DEFAULT_COLOR } from "@/lib/constants";

interface TrainingData {
  [algo: string]: { timesteps: number[]; rewards: number[] };
}

export function TrainingCurves() {
  const { data, loading, error } = useArtifact<TrainingData>("/artifacts/reward-curves");

  if (loading) return <p className="text-zinc-400 text-sm">Loading training curves…</p>;
  if (error)   return <p className="text-red-400 text-sm">Error: {error}</p>;
  if (!data)   return null;

  // Merge all algorithms into a single timestep-indexed array
  const allTimesteps = Array.from(
    new Set(Object.values(data).flatMap((d) => d.timesteps))
  ).sort((a, b) => a - b);

  const chartData = allTimesteps.map((t) => {
    const point: Record<string, number | undefined> = { timestep: t };
    Object.entries(data).forEach(([algo, d]) => {
      const idx = d.timesteps.indexOf(t);
      point[algo] = idx >= 0 ? d.rewards[idx] : undefined;
    });
    return point;
  });

  const algos = Object.keys(data);

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            dataKey="timestep"
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
            tick={{ fontSize: 11 }}
            label={{ value: "Environment Steps", position: "insideBottomRight", offset: -5 }}
          />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "Mean Episode Reward", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(v: number) => v.toFixed(1)} labelFormatter={(v: number) => `${(v / 1000).toFixed(0)}k steps`} />
          <Legend />
          {algos.map((algo) => (
            <Line
              key={algo}
              type="monotone"
              dataKey={algo}
              stroke={ALGO_COLORS[algo] ?? DEFAULT_COLOR}
              dot={false}
              strokeWidth={2}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

## Step 8 — `components/rollout-error-chart.tsx`

Horizon vs. MSE plot. This is Fig 2 from Phase 4.

```tsx
// dashboard/components/rollout-error-chart.tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { useArtifact } from "@/hooks/use-artifact";
import { ALGO_COLORS, DEFAULT_COLOR } from "@/lib/constants";

interface RolloutErrors {
  [algo: string]: number[];   // array indexed 0..H-1
}

export function RolloutErrorChart() {
  const { data, loading, error } = useArtifact<RolloutErrors>("/artifacts/rollout-errors");

  if (loading) return <p className="text-zinc-400 text-sm">Loading rollout error data…</p>;
  if (error)   return <p className="text-red-400 text-sm">Error: {error}</p>;
  if (!data)   return null;

  const maxHorizon = Math.max(...Object.values(data).map((a) => a.length));
  const chartData = Array.from({ length: maxHorizon }, (_, i) => {
    const point: Record<string, number> = { horizon: i + 1 };
    Object.entries(data).forEach(([algo, errors]) => {
      if (errors[i] !== undefined) point[algo] = errors[i];
    });
    return point;
  });

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis dataKey="horizon" label={{ value: "Prediction Horizon (steps)", position: "insideBottom", offset: -5 }} tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "Latent MSE", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <Legend />
          {Object.keys(data).map((algo) => (
            <Line
              key={algo}
              type="monotone"
              dataKey={algo}
              stroke={ALGO_COLORS[algo] ?? DEFAULT_COLOR}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

---

## Step 9 — `components/comparison-table.tsx`

Renders `comparison_table.csv` as a sortable HTML table.

```tsx
// dashboard/components/comparison-table.tsx
"use client";
import { useArtifact } from "@/hooks/use-artifact";

interface Row {
  algorithm: string;
  mean_reward: string;
  std_reward: string;
  [key: string]: string;
}

export function ComparisonTable() {
  const { data, loading, error } = useArtifact<Row[]>("/artifacts/comparison-table");

  if (loading) return <p className="text-zinc-400 text-sm">Loading comparison table…</p>;
  if (error)   return <p className="text-red-400 text-sm">Error: {error}</p>;
  if (!data || data.length === 0) return <p className="text-zinc-500 text-sm">No data yet — run Phase 4 evaluation.</p>;

  const columns = Object.keys(data[0]);

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-700">
      <table className="min-w-full text-sm">
        <thead className="bg-zinc-800 text-zinc-300">
          <tr>
            {columns.map((col) => (
              <th key={col} className="px-4 py-2 text-left font-medium capitalize">
                {col.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-700">
          {data.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-zinc-900" : "bg-zinc-800/50"}>
              {columns.map((col) => (
                <td key={col} className="px-4 py-2 text-zinc-200">
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
```

---

## Step 10 — `components/nav.tsx`

Tab navigation shared across all pages.

```tsx
// dashboard/components/nav.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/live",     label: "Live Rollout" },
  { href: "/training", label: "Training Curves" },
  { href: "/analysis", label: "Rollout Error & Efficiency" },
  { href: "/results",  label: "Comparison Table" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 border-b border-zinc-700 px-4 pt-3 bg-zinc-950">
      {TABS.map(({ href, label }) => {
        const active = pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              active
                ? "bg-zinc-800 text-white border border-b-0 border-zinc-700"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
```

---

## Step 11 — App Layout and Pages

### `app/layout.tsx`

```tsx
// dashboard/app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RL Training Dashboard",
  description: "TD-MPC2 + SSM Evaluation Dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-zinc-950 text-zinc-100 min-h-screen`}>
        <header className="px-6 pt-4 pb-1">
          <h1 className="text-xl font-bold tracking-tight">
            Efficient MBRL with SSM World Models
            <span className="ml-3 text-sm font-normal text-zinc-400">
              TD-MPC2 · S4 · S5 · Mamba
            </span>
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5">
            Riddhi Poddar · Vedant Shinde · Soham Kulkarni · Sri Vaishnavi Kodali
          </p>
        </header>
        <Nav />
        <main className="px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
```

### `app/page.tsx` (redirect to /live)

```tsx
// dashboard/app/page.tsx
import { redirect } from "next/navigation";
export default function Root() { redirect("/live"); }
```

### `app/live/page.tsx`

```tsx
// dashboard/app/live/page.tsx
"use client";
import { useWebSocket } from "@/hooks/use-websocket";
import { VideoPanel } from "@/components/video-panel";
import { LiveRewardChart } from "@/components/live-reward-chart";

export default function LivePage() {
  const { message, connected } = useWebSocket();

  return (
    <div className="space-y-6">
      {/* Connection status */}
      <div className="flex items-center gap-2 text-sm">
        <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
        <span className="text-zinc-400">{connected ? "Connected to backend" : "Reconnecting…"}</span>
      </div>

      {/* Video grid */}
      {message ? (
        <div className="grid gap-4" style={{
          gridTemplateColumns: `repeat(${Math.min(message.labels.length, 3)}, 1fr)`
        }}>
          {message.labels.map((label, i) => (
            <VideoPanel
              key={label}
              label={label}
              frame={message.frames[i] ?? ""}
              metric={message.metrics[i]}
            />
          ))}
        </div>
      ) : (
        <div className="text-zinc-500 text-sm">
          {connected ? "Waiting for first frame…" : "Backend offline — start the FastAPI server."}
        </div>
      )}

      {/* Live rolling reward */}
      {message && (
        <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">Live Step Reward</h2>
          <LiveRewardChart
            labels={message.labels}
            latestMetrics={message.metrics}
          />
        </div>
      )}
    </div>
  );
}
```

### `app/training/page.tsx`

```tsx
// dashboard/app/training/page.tsx
import { TrainingCurves } from "@/components/training-curves";

export default function TrainingPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-zinc-100">Fig 1 — Training Reward Curves</h2>
        <p className="text-sm text-zinc-400 mt-1">
          All algorithms — reward vs. environment steps. Loaded from{" "}
          <code className="text-zinc-300">artifacts/*/metrics.jsonl</code>.
        </p>
      </div>
      <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4">
        <TrainingCurves />
      </div>
    </div>
  );
}
```

### `app/analysis/page.tsx`

```tsx
// dashboard/app/analysis/page.tsx
import { RolloutErrorChart } from "@/components/rollout-error-chart";

export default function AnalysisPage() {
  return (
    <div className="space-y-8">
      {/* Fig 2 */}
      <div className="space-y-3">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">Fig 2 — Multi-Step Rollout Error</h2>
          <p className="text-sm text-zinc-400 mt-1">
            Latent MSE at each prediction horizon. Loaded from{" "}
            <code className="text-zinc-300">artifacts/*/rollout_errors.npy</code>.
          </p>
        </div>
        <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4">
          <RolloutErrorChart />
        </div>
      </div>

      {/* Fig 4 note */}
      <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-4 text-sm text-zinc-400">
        <strong className="text-zinc-200">Fig 4 — Sample Efficiency</strong> is derived from the
        training curves on the <em>Training Curves</em> tab — read reward values at 50k, 100k, and
        200k steps from the chart above. A dedicated bar chart view can be added by extending{" "}
        <code className="text-zinc-300">components/training-curves.tsx</code> with checkpoint markers.
      </div>
    </div>
  );
}
```

### `app/results/page.tsx`

```tsx
// dashboard/app/results/page.tsx
import { ComparisonTable } from "@/components/comparison-table";

export default function ResultsPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-zinc-100">Final Comparison Table</h2>
        <p className="text-sm text-zinc-400 mt-1">
          Loaded from <code className="text-zinc-300">artifacts/comparison_table.csv</code>.
          Generated by <code className="text-zinc-300">evaluation/eval_runner.py</code>.
        </p>
      </div>
      <ComparisonTable />
    </div>
  );
}
```

---

## Step 12 — Environment Variables

Create `dashboard/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

For production / demo mode on a different host, update these two lines only.

---

## Step 13 — Running the Dashboard

```bash
cd rl/dashboard
npm run dev
```

Open `http://localhost:3000`. The dashboard auto-reconnects to the backend WebSocket every 2 seconds if the connection drops.

---

## Full Build Order

Follow this order exactly — each step depends on the previous one being working:

1. Start the FastAPI backend (`uvicorn server.server:app --reload --port 8000`)
2. Verify `http://localhost:8000/health` returns `{"status":"ok"}`
3. Scaffold Next.js project (`npx create-next-app`)
4. Add `lib/constants.ts` and both hooks (`use-websocket.ts`, `use-artifact.ts`)
5. Build `video-panel.tsx` — test with `http://localhost:3000/live` showing frames
6. Build `live-reward-chart.tsx` — confirm reward updates every frame
7. Build `training-curves.tsx` — test `http://localhost:3000/training`
8. Build `rollout-error-chart.tsx` — test `http://localhost:3000/analysis`
9. Build `comparison-table.tsx` — test `http://localhost:3000/results`
10. Polish: add loading spinners, error states, and `Nav`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `WebSocket connection refused` | Backend not running — start `uvicorn` first |
| Videos show blank grey boxes | Backend loaded but no checkpoints found — check `artifacts/` paths |
| Charts show "Error: HTTP 404" | REST endpoint not registered — confirm `server.py` has the route |
| CORS error in browser | Add `http://localhost:3000` to `allow_origins` in `server.py` |
| Frames freeze after a few seconds | Reduce `STREAM_FPS` in `config.py` or `MPPI_SAMPLES` for slower machines |
| `rollout_errors.npy` not found | Phase 2/3 training must complete before this chart has data |
| `comparison_table.csv` empty | Run `evaluation/eval_runner.py` from Phase 4 first |
