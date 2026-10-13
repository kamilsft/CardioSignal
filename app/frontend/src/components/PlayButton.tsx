import { useEffect, useRef, useState } from "react";
import { isAudioSupported, playSignal, type Playback } from "../lib/audio";

interface Props {
  signal: readonly number[];
  sampleRate: number;
  label: string;
}

export function PlayButton({ signal, sampleRate, label }: Props) {
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const playback = useRef<Playback | null>(null);

  // Stop when the signal changes (a new result replaced the old one) or on unmount.
  useEffect(() => () => {
    playback.current?.stop();
    playback.current = null;
    setPlaying(false);
  }, [signal]);

  if (!isAudioSupported()) return null;

  function toggle() {
    if (playing) {
      playback.current?.stop();
      return;
    }
    try {
      const p = playSignal(signal, sampleRate);
      playback.current = p;
      setPlaying(true);
      setError(null);
      void p.finished.then(() => setPlaying(false));
    } catch {
      setError("Could not play audio in this browser.");
    }
  }

  return (
    <span className="play">
      <button type="button" className="secondary" onClick={toggle} aria-pressed={playing}>
        <span aria-hidden="true">{playing ? "■" : "▶"}</span> {playing ? "Stop" : label}
      </button>
      {error && <small className="form-error">{error}</small>}
    </span>
  );
}
