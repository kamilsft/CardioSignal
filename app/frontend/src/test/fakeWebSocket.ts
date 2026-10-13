/** Minimal stand-in for the browser WebSocket, driven by tests. */
export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static readonly OPEN = 1;
  static readonly CLOSED = 3;

  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  static latest(): FakeWebSocket {
    return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  }

  static reset(): void {
    FakeWebSocket.instances = [];
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  receive(message: unknown): void {
    this.onmessage?.({ data: typeof message === "string" ? message : JSON.stringify(message) });
  }

  close(): void {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  send(): void {}
}

export function snapshot(overrides: Record<string, unknown> = {}) {
  return {
    type: "snapshot",
    time: 1,
    summary: { patients: 2, devices_connected: 2, ok: 1, warming_up: 0, no_signal: 1, analysis_ms: 3.2 },
    patients: [
      { id: "patient-0001", heart_rate_bpm: 72.4, status: "ok" },
      { id: "patient-0002", heart_rate_bpm: null, status: "no_signal" },
    ],
    waveforms: { "patient-0001": { fs: 125, samples: [0, 0.5, 1, 0.5, 0] } },
    ...overrides,
  };
}
