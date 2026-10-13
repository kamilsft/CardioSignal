export interface MinMaxBucket {
  min: number;
  max: number;
}

/**
 * Reduce a long signal to `buckets` (min, max) pairs so a chart can draw it
 * pixel-accurately without plotting every sample. Spikes are never lost.
 */
export function minMaxDecimate(signal: readonly number[], buckets: number): MinMaxBucket[] {
  if (signal.length === 0 || buckets <= 0) return [];
  const count = Math.min(buckets, signal.length);
  const size = signal.length / count;
  return Array.from({ length: count }, (_, b) => {
    const start = Math.floor(b * size);
    const end = Math.max(start + 1, Math.floor((b + 1) * size));
    let min = Infinity;
    let max = -Infinity;
    for (let i = start; i < end; i++) {
      const v = signal[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  });
}

/** Linear-interpolation resampling. Returns a new array. */
export function resampleLinear(signal: readonly number[], fromRate: number, toRate: number): Float32Array {
  if (signal.length === 0) return new Float32Array(0);
  const length = Math.max(1, Math.round((signal.length * toRate) / fromRate));
  const out = new Float32Array(length);
  const step = fromRate / toRate;
  for (let i = 0; i < length; i++) {
    const pos = i * step;
    const left = Math.min(Math.floor(pos), signal.length - 1);
    const right = Math.min(left + 1, signal.length - 1);
    const frac = pos - left;
    out[i] = signal[left] * (1 - frac) + signal[right] * frac;
  }
  return out;
}

/** Scale so the loudest sample hits `peak`, keeping playback loud but unclipped. */
export function normalizePeak(signal: Float32Array, peak = 0.9): Float32Array {
  let maxAbs = 0;
  for (const v of signal) maxAbs = Math.max(maxAbs, Math.abs(v));
  if (maxAbs === 0) return new Float32Array(signal);
  const gain = peak / maxAbs;
  return signal.map((v) => v * gain);
}

export function formatBpm(bpm: number | null | undefined): string {
  return bpm == null ? "—" : `${Math.round(bpm)} bpm`;
}

export function durationSeconds(samples: number, sampleRate: number): number {
  return sampleRate > 0 ? samples / sampleRate : 0;
}
