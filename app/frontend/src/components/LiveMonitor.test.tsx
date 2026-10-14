import { act, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeWebSocket, snapshot } from "../test/fakeWebSocket";
import { LiveMonitor } from "./LiveMonitor";

beforeEach(() => {
  FakeWebSocket.reset();
  vi.stubGlobal("WebSocket", FakeWebSocket);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function connect() {
  render(<LiveMonitor />);
  const ws = FakeWebSocket.latest();
  act(() => ws.open());
  return ws;
}

describe("LiveMonitor", () => {
  it("connects to the live stream endpoint", () => {
    render(<LiveMonitor />);
    expect(FakeWebSocket.latest().url).toMatch(/^ws:\/\/.+\/api\/live\/stream$/);
    expect(screen.getByRole("status")).toHaveTextContent("Connecting");
  });

  it("shows heart rate and status for each patient", () => {
    const ws = connect();
    act(() => ws.receive(snapshot()));
    const tile = screen.getByRole("article", { name: /patient-0001/ });
    expect(within(tile).getByText("72")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Live");
    expect(screen.getAllByText("No signal").length).toBeGreaterThan(0);
  });

  it("shows the summary counts", () => {
    const ws = connect();
    act(() => ws.receive(snapshot()));
    const summary = screen.getByRole("region", { name: "Summary" });
    expect(within(summary).getByText("Patients").previousSibling).toHaveTextContent("2");
    expect(within(summary).getByText("Analysis time per tick").previousSibling).toHaveTextContent("3 ms");
  });

  it("explains how to start the simulator when nobody is streaming", () => {
    const ws = connect();
    act(() => ws.receive(snapshot({ patients: [], waveforms: {},
      summary: { patients: 0, devices_connected: 0, ok: 0, warming_up: 0, no_signal: 0, analysis_ms: 0 } })));
    expect(screen.getByText(/No patients are streaming/)).toBeInTheDocument();
    expect(screen.getByText(/cardiosignal_api.live.simulator/)).toBeInTheDocument();
  });

  it("ignores malformed messages", () => {
    const ws = connect();
    act(() => ws.receive("not json"));
    act(() => ws.receive({ type: "snapshot", patients: "nope" }));
    expect(screen.getByRole("status")).toHaveTextContent("Live");
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });

  it("reconnects after the connection drops", () => {
    vi.useFakeTimers();
    const ws = connect();
    act(() => ws.close());
    expect(screen.getByRole("status")).toHaveTextContent("Reconnecting");
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("closes the connection and stops reconnecting when unmounted", () => {
    vi.useFakeTimers();
    const { unmount } = render(<LiveMonitor />);
    const ws = FakeWebSocket.latest();
    act(() => ws.open());
    unmount();
    expect(ws.readyState).toBe(FakeWebSocket.CLOSED);
    act(() => vi.advanceTimersByTime(30_000));
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});
