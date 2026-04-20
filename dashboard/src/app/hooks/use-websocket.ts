import { useEffect, useRef, useState, useCallback } from "react";
import { getWebSocketUrl } from "../lib/api";

export interface StepMetric {
  label: string;
  run_name?: string;
  display_name?: string;
  step: number;
  reward: number;
  episode_reward: number;
  done: boolean;
  action_magnitude: number;
}

export interface LiveModelCard {
  label: string;
  run_name: string;
  display_name: string;
  algorithm_name: string;
  env_title: string;
  behavior_text: string;
}

export interface WSMessage {
  labels: string[];
  models?: LiveModelCard[];
  frames: string[];
  metrics: StepMetric[];
}

export function useWebSocket() {
  const [message, setMessage] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(getWebSocketUrl("/ws"));
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
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
