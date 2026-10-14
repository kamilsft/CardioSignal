import { useEffect, useRef } from "react";
import type { PatientStatus, PatientVitals } from "../api/liveSchemas";
import type { Waveform } from "../hooks/useLiveStream";
import { resolveColor } from "../lib/cssColor";

const HEIGHT = 64;

export const STATUS_LABEL: Record<PatientStatus, string> = {
  ok: "OK",
  warming_up: "Connecting…",
  no_signal: "No signal",
};

interface Props {
  patient: PatientVitals;
  waveform?: Waveform;
  windowSeconds: number;
}

function drawTrace(canvas: HTMLCanvasElement, samples: readonly number[], capacity: number) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  if (!ctx || width === 0) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(HEIGHT * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, HEIGHT);
  if (samples.length < 2) return;

  let peak = 1e-6;
  for (const v of samples) peak = Math.max(peak, Math.abs(v));
  const step = width / Math.max(1, capacity - 1);
  const offset = (capacity - samples.length) * step;     // newest sample at the right edge
  ctx.strokeStyle = resolveColor(canvas, "var(--ecg)");
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  samples.forEach((v, i) => {
    const x = offset + i * step;
    const y = HEIGHT / 2 - (v / peak) * (HEIGHT / 2 - 4);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export function PatientTile({ patient, waveform, windowSeconds }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hr = patient.heart_rate_bpm;

  useEffect(() => {
    if (canvasRef.current && waveform) {
      drawTrace(canvasRef.current, waveform.samples, Math.round(waveform.fs * windowSeconds));
    }
  }, [waveform, windowSeconds]);

  const label = hr !== null ? `${Math.round(hr)} bpm` : STATUS_LABEL[patient.status];
  return (
    <article className={`patient-tile status-${patient.status}`} aria-label={`${patient.id}: ${label}`}>
      <header className="patient-head">
        <span className="patient-id">{patient.id}</span>
        <span className={`status-badge status-${patient.status}`}>{STATUS_LABEL[patient.status]}</span>
      </header>
      <canvas ref={canvasRef} className="patient-trace" style={{ height: HEIGHT }} aria-hidden="true" />
      <p className="patient-hr">
        {hr !== null ? (
          <><span className="hr-value">{Math.round(hr)}</span> <span className="hr-unit">bpm</span></>
        ) : (
          <span className="hr-missing">—</span>
        )}
      </p>
    </article>
  );
}
