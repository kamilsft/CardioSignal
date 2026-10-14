import { describe, expect, it } from "vitest";
import type { PatientVitals } from "../api/liveSchemas";
import { badgeFor, classifyHeartRate } from "./PatientTile";

describe("classifyHeartRate", () => {
  it.each([
    [39, "critical", "Very slow"],
    [41, "slow", "Slow"],
    [59, "slow", "Slow"],
    [60, "normal", "Normal"],
    [100, "normal", "Normal"],
    [101, "fast", "Fast"],
    [151, "critical", "Very fast"],
  ])("%s bpm is %s", (bpm, rhythm, label) => {
    expect(classifyHeartRate(bpm)).toEqual({ rhythm, label });
  });
});

describe("badgeFor", () => {
  const patient = (overrides: Partial<PatientVitals>) => ({ id: "p", heart_rate_bpm: 72, status: "ok", ...overrides }) as PatientVitals;

  it("shows the rhythm, not 'OK', when there is a heart rate", () => {
    expect(badgeFor(patient({ heart_rate_bpm: 41 }))).toEqual({ className: "rhythm-slow", label: "Slow" });
  });

  it("classifies the rounded value the tile shows", () => {
    expect(badgeFor(patient({ heart_rate_bpm: 59.6 })).label).toBe("Normal");   // displayed as 60 bpm
  });

  it("falls back to the connection status without a reading", () => {
    expect(badgeFor(patient({ heart_rate_bpm: null, status: "no_signal" }))).toEqual({ className: "status-no_signal", label: "No signal" });
  });
});
