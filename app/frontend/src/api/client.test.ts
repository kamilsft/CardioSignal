import { afterEach, describe, expect, it, vi } from "vitest";
import { jsonResponse, ok, SAMPLES, translation } from "../test/fixtures";
import { ApiError, listSamples, translateFile, translateSignal } from "./client";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

afterEach(() => fetchMock.mockReset());

describe("api client", () => {
  it("returns validated data from the envelope", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(ok(SAMPLES)));
    await expect(listSamples()).resolves.toEqual(SAMPLES);
    expect(fetchMock).toHaveBeenCalledWith("/api/samples", undefined);
  });

  it("surfaces the server's error message", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ success: false, data: null, error: "Signal is 1.0 s long" }, 422));
    await expect(translateSignal([1], 2000, "ecg_to_pcg")).rejects.toThrow("Signal is 1.0 s long");
  });

  it("rejects responses with an unexpected shape", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(ok({ direction: "sideways" })));
    await expect(translateSignal([1], 2000, "ecg_to_pcg")).rejects.toThrow("expected format");
  });

  it("explains network failures", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(listSamples()).rejects.toBeInstanceOf(ApiError);
  });

  it("handles non-JSON error pages", async () => {
    fetchMock.mockResolvedValueOnce(new Response("<html>Bad gateway</html>", { status: 502 }));
    await expect(listSamples()).rejects.toThrow("HTTP 502");
  });

  it("explains nginx's too-large page", async () => {
    fetchMock.mockResolvedValueOnce(new Response("<html>413</html>", { status: 413 }));
    await expect(listSamples()).rejects.toThrow("too large");
  });

  it("sends uploads as multipart form data", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(ok(translation())));
    await translateFile(new File(["1\n2"], "ecg.csv"), "ecg_to_pcg", 500);
    const [url, init] = fetchMock.mock.calls[0];
    const form = init.body as FormData;
    expect(url).toBe("/api/translate/upload");
    expect(form.get("direction")).toBe("ecg_to_pcg");
    expect(form.get("sample_rate")).toBe("500");
  });
});
