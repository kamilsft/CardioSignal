import { useEffect, useState } from "react";
import { liveStreamUrl } from "../api/client";
import { liveSnapshotSchema, type LiveSummary, type PatientVitals } from "../api/liveSchemas";
import { appendRolling } from "../lib/rolling";

export type Connection = "connecting" | "live" | "reconnecting";

export interface Waveform {
  fs: number;
  samples: readonly number[];
}

export interface LiveState {
  connection: Connection;
  summary: LiveSummary | null;
  patients: PatientVitals[];
  waveforms: Readonly<Record<string, Waveform>>;
}

const INITIAL: LiveState = { connection: "connecting", summary: null, patients: [], waveforms: {} };
const RETRY_START_MS = 1000;
const RETRY_MAX_MS = 10_000;

function parse(data: unknown) {
  if (typeof data !== "string") return null;
  try {
    const result = liveSnapshotSchema.safeParse(JSON.parse(data));
    return result.success ? result.data : null;
  } catch {
    return null;
  }
}

function mergeWaveforms(prev: LiveState["waveforms"], incoming: Record<string, Waveform>, seconds: number) {
  const next: Record<string, Waveform> = {};
  for (const [id, chunk] of Object.entries(incoming)) {
    const previous = prev[id]?.fs === chunk.fs ? prev[id].samples : [];
    next[id] = { fs: chunk.fs, samples: appendRolling(previous, chunk.samples, Math.round(chunk.fs * seconds)) };
  }
  return next;    // patients no longer sent are dropped
}

/** Subscribe to the live monitor; reconnects with backoff and keeps `windowSeconds` of waveform per patient. */
export function useLiveStream(windowSeconds = 6, url: string = liveStreamUrl()): LiveState {
  const [state, setState] = useState<LiveState>(INITIAL);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let retryMs = RETRY_START_MS;
    let stopped = false;

    function scheduleRetry() {
      setState((s) => ({ ...s, connection: "reconnecting" }));
      retryTimer = setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, RETRY_MAX_MS);
    }

    function connect() {
      try {
        socket = new WebSocket(url);
      } catch {
        scheduleRetry();      // e.g. the browser blocked the URL; keep trying like any other failure
        return;
      }
      socket.onopen = () => {
        retryMs = RETRY_START_MS;
        setState((s) => ({ ...s, connection: "live" }));
      };
      socket.onmessage = (event: MessageEvent) => {
        const snapshot = parse(event.data);
        if (!snapshot) return;
        setState((s) => ({
          connection: "live",
          summary: snapshot.summary,
          patients: snapshot.patients,
          waveforms: mergeWaveforms(s.waveforms, snapshot.waveforms, windowSeconds),
        }));
      };
      socket.onclose = () => {
        if (!stopped) scheduleRetry();
      };
    }

    connect();
    return () => {
      stopped = true;
      clearTimeout(retryTimer);
      socket?.close();
    };
  }, [url, windowSeconds]);

  return state;
}
