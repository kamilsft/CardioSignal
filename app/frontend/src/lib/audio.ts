import { normalizePeak, resampleLinear } from "./signal";

/**
 * Heart sounds live at 2 kHz in the model, but browsers only play buffers at
 * common audio rates, so we upsample before playback.
 */
export const PLAYBACK_RATE = 22050;

export function prepareForPlayback(signal: readonly number[], sampleRate: number): Float32Array {
  return normalizePeak(resampleLinear(signal, sampleRate, PLAYBACK_RATE));
}

type AudioContextCtor = typeof AudioContext;

function audioContextCtor(): AudioContextCtor | null {
  const w = window as unknown as { AudioContext?: AudioContextCtor; webkitAudioContext?: AudioContextCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export function isAudioSupported(): boolean {
  return audioContextCtor() !== null;
}

export interface Playback {
  stop: () => void;
  finished: Promise<void>;
}

export function playSignal(signal: readonly number[], sampleRate: number): Playback {
  const Ctor = audioContextCtor();
  if (!Ctor) throw new Error("Audio playback is not supported in this browser.");
  const context = new Ctor();
  const samples = prepareForPlayback(signal, sampleRate);
  const buffer = context.createBuffer(1, samples.length, PLAYBACK_RATE);
  buffer.copyToChannel(samples, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  const finished = new Promise<void>((resolve) => {
    source.onended = () => {
      void context.close();
      resolve();
    };
  });
  source.start();
  return { stop: () => source.stop(), finished };
}
