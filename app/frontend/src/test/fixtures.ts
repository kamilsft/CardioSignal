import type { Sample, SampleSummary, Translation } from "../api/schemas";

export const SAMPLES: SampleSummary[] = [
  { id: "healthy-1", label: "healthy", description: "Healthy patient 1 (unseen patient)" },
  { id: "cvd-1", label: "cvd", description: "Cardiovascular disease patient 1 (unseen patient)" },
];

const wave = (n: number) => Array.from({ length: n }, (_, i) => Math.sin(i / 50) * 0.5);

export const SAMPLE: Sample = { ...SAMPLES[0], sample_rate: 2000, ecg: wave(12000), pcg: wave(12000) };

export function translation(overrides: Partial<Translation> = {}): Translation {
  return {
    direction: "ecg_to_pcg",
    sample_rate: 2000,
    input_signal: wave(12000),
    output_signal: wave(12000),
    input_analysis: { heart_rate_bpm: 72.4, beats: [100, 1766], quality: null, quality_ok: null },
    output_analysis: { heart_rate_bpm: 71.9, beats: [], quality: 0.62, quality_ok: true },
    ecg_was_flipped: false,
    warnings: [],
    ...overrides,
  };
}

export function ok<T>(data: T) {
  return { success: true, data, error: null };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
