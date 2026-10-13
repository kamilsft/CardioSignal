import { useState, type FormEvent } from "react";

const ACCEPTED = ".csv,.txt,.npy,.wav";
const MAX_BYTES = 5 * 1024 * 1024;

interface Props {
  signalName: string;
  onSubmit: (file: File, sampleRate: number | null) => void;
  disabled?: boolean;
}

export function validateUpload(file: File | null, sampleRateText: string): { error: string | null; sampleRate: number | null } {
  if (!file) return { error: "Choose a file first.", sampleRate: null };
  if (file.size > MAX_BYTES) return { error: "File is too large (limit 5 MB).", sampleRate: null };
  if (file.name.toLowerCase().endsWith(".wav")) return { error: null, sampleRate: null };
  const rate = Number(sampleRateText);
  if (!Number.isInteger(rate) || rate < 250 || rate > 48000) {
    return { error: "Enter the sample rate in Hz (a whole number between 250 and 48000).", sampleRate: null };
  }
  return { error: null, sampleRate: rate };
}

export function UploadForm({ signalName, onSubmit, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sampleRate, setSampleRate] = useState("");
  const [error, setError] = useState<string | null>(null);
  const isWav = file?.name.toLowerCase().endsWith(".wav") ?? false;

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const result = validateUpload(file, sampleRate);
    setError(result.error);
    if (!result.error && file) onSubmit(file, result.sampleRate);
  }

  return (
    <form className="upload-form" onSubmit={handleSubmit} noValidate>
      <label className="field">
        <span>{signalName} file</span>
        <input type="file" accept={ACCEPTED} disabled={disabled}
          onChange={(e) => { setFile(e.target.files?.[0] ?? null); setError(null); }} />
        <small className="muted">One column of numbers (CSV/TXT), a NumPy .npy array, or a .wav file. 3–60 seconds.</small>
      </label>
      {!isWav && (
        <label className="field field-narrow">
          <span>Sample rate (Hz)</span>
          <input type="number" inputMode="numeric" min={250} max={48000} placeholder="e.g. 500" value={sampleRate}
            disabled={disabled} onChange={(e) => { setSampleRate(e.target.value); setError(null); }} />
        </label>
      )}
      {error && <p className="form-error" role="alert">{error}</p>}
      <button type="submit" className="primary" disabled={disabled}>Translate file</button>
    </form>
  );
}
