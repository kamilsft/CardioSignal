import { describe, expect, it } from "vitest";
import { prepareForPlayback, PLAYBACK_RATE } from "./audio";
import { durationSeconds, formatBpm, minMaxDecimate, normalizePeak, resampleLinear } from "./signal";

describe("minMaxDecimate", () => {
  it("keeps spikes that a naive stride would drop", () => {
    const signal = new Array(1000).fill(0);
    signal[503] = 1;
    const buckets = minMaxDecimate(signal, 10);
    expect(buckets).toHaveLength(10);
    expect(buckets[5]).toEqual({ min: 0, max: 1 });
    expect(Math.max(...buckets.map((b) => b.max))).toBe(1);
  });

  it("handles short and empty input", () => {
    expect(minMaxDecimate([1, 2], 10)).toEqual([{ min: 1, max: 1 }, { min: 2, max: 2 }]);
    expect(minMaxDecimate([], 10)).toEqual([]);
  });
});

describe("resampleLinear", () => {
  it("changes length by the rate ratio and interpolates", () => {
    const out = resampleLinear([0, 1, 2, 3], 2, 4);
    expect(out).toHaveLength(8);
    expect(out[1]).toBeCloseTo(0.5);
    expect(out[7]).toBeCloseTo(3);
  });

  it("does not modify its input", () => {
    const input = [1, 2, 3];
    resampleLinear(input, 1, 2);
    expect(input).toEqual([1, 2, 3]);
  });
});

describe("normalizePeak", () => {
  it("scales to the target peak and leaves silence alone", () => {
    expect(Math.max(...normalizePeak(new Float32Array([0.1, -0.2]), 0.8).map(Math.abs))).toBeCloseTo(0.8);
    expect([...normalizePeak(new Float32Array([0, 0]))]).toEqual([0, 0]);
  });
});

describe("helpers", () => {
  it("formats heart rate and duration", () => {
    expect(formatBpm(71.6)).toBe("72 bpm");
    expect(formatBpm(null)).toBe("—");
    expect(durationSeconds(12000, 2000)).toBe(6);
    expect(durationSeconds(10, 0)).toBe(0);
  });

  it("prepares heart sounds for browser playback", () => {
    const out = prepareForPlayback(new Array(2000).fill(0).map((_, i) => Math.sin(i / 10) * 0.2), 2000);
    expect(out.length).toBe(PLAYBACK_RATE);
    expect(Math.max(...out)).toBeCloseTo(0.9, 1);
  });
});
