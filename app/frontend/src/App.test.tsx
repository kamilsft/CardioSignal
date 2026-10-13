import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { FakeWebSocket } from "./test/fakeWebSocket";
import { jsonResponse, ok, SAMPLE, SAMPLES, translation } from "./test/fixtures";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

function route(handlers: Record<string, () => Response>) {
  fetchMock.mockImplementation(async (url: string) => {
    const key = Object.keys(handlers).find((k) => url.endsWith(k));
    return key ? handlers[key]() : jsonResponse({ success: false, error: "not found" }, 404);
  });
}

beforeEach(() => {
  route({
    "/api/samples": () => jsonResponse(ok(SAMPLES)),
    "/api/samples/healthy-1": () => jsonResponse(ok(SAMPLE)),
    "/api/translate": () => jsonResponse(ok(translation())),
  });
});
afterEach(() => fetchMock.mockReset());

describe("App", () => {
  it("translates a demo recording and shows the results", async () => {
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: /Healthy patient 1/ }));
    expect(await screen.findByText("Generated heart sound")).toBeInTheDocument();
    expect(screen.getAllByText("72 bpm")).toHaveLength(2); // input and generated heart rate
    expect(screen.getByText("Input ECG")).toBeInTheDocument();
    const body = JSON.parse(fetchMock.mock.calls.find(([u]) => u === "/api/translate")![1].body);
    expect(body.direction).toBe("ecg_to_pcg");
    expect(body.signal).toHaveLength(12000);
  });

  it("re-translates the selected sample when the direction changes", async () => {
    route({
      "/api/samples": () => jsonResponse(ok(SAMPLES)),
      "/api/samples/healthy-1": () => jsonResponse(ok(SAMPLE)),
      "/api/translate": () => jsonResponse(ok(translation({
        direction: "pcg_to_ecg",
        input_analysis: { heart_rate_bpm: 70, beats: [], quality: 0.2, quality_ok: false },
        output_analysis: { heart_rate_bpm: 69, beats: [10], quality: null, quality_ok: null },
        warnings: ["The heart-sound recording has a weak or irregular rhythm (low quality)."],
      }))),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: /Healthy patient 1/ }));
    await userEvent.click(screen.getByRole("radio", { name: "Heart sound → ECG" }));
    expect(await screen.findByText("Generated ECG")).toBeInTheDocument();
    expect(screen.getByText(/weak or irregular rhythm/)).toBeInTheDocument();
    expect(screen.getByText(/weak rhythm: low confidence/)).toBeInTheDocument();
  });

  it("shows server errors to the user", async () => {
    route({
      "/api/samples": () => jsonResponse(ok(SAMPLES)),
      "/api/samples/healthy-1": () => jsonResponse(ok(SAMPLE)),
      "/api/translate": () => jsonResponse({ success: false, data: null, error: "Too many requests." }, 429),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: /Healthy patient 1/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many requests.");
  });

  it("switches to the live monitor and back", async () => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: "Live monitor" }));
    expect(await screen.findByRole("heading", { name: "Live monitor" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Live monitor" })).toHaveAttribute("aria-current", "page");
    await userEvent.click(screen.getByRole("link", { name: "Translate" }));
    expect(await screen.findByRole("heading", { name: "CardioSignal" })).toBeInTheDocument();
    window.location.hash = "";
  });

  it("reports when the backend is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Could not reach"));
  });
});
