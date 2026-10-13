import { useEffect, useRef, useState } from "react";
import { resolveColor } from "../lib/cssColor";
import { minMaxDecimate } from "../lib/signal";

export interface Series {
  label: string;
  values: readonly number[];
  color: string; // CSS color or var(--token)
  faded?: boolean;
}

interface Props {
  series: Series[];
  sampleRate: number;
  beats?: readonly number[];
  windowSeconds?: number;
  height?: number;
  title: string;
}

const PADDING = { top: 8, right: 8, bottom: 22, left: 8 };

function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

export function SignalChart({ series, sampleRate, beats = [], windowSeconds = 6, height = 150, title }: Props) {
  const [wrapperRef, width] = useElementWidth<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [offset, setOffset] = useState(0);
  const total = Math.max(0, ...series.map((s) => s.values.length));
  const windowSamples = Math.min(total, Math.round(windowSeconds * sampleRate));
  const maxOffset = Math.max(0, total - windowSamples);
  const start = Math.min(offset, maxOffset);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const plotW = width - PADDING.left - PADDING.right;
    const plotH = height - PADDING.top - PADDING.bottom;
    const y = (v: number) => PADDING.top + ((1 - v) / 2) * plotH; // signals live in [-1, 1]
    const xOf = (sample: number) => PADDING.left + ((sample - start) / Math.max(1, windowSamples)) * plotW;

    ctx.strokeStyle = resolveColor(canvas, "var(--grid)");
    ctx.fillStyle = resolveColor(canvas, "var(--text-muted)");
    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, y(0));
    ctx.lineTo(width - PADDING.right, y(0));
    ctx.stroke();
    for (let s = Math.ceil(start / sampleRate); s * sampleRate <= start + windowSamples; s++) {
      const x = xOf(s * sampleRate);
      ctx.fillText(`${s}s`, Math.min(x, width - 20), height - 6);
    }

    ctx.strokeStyle = resolveColor(canvas, "var(--beat)");
    ctx.setLineDash([3, 3]);
    for (const b of beats) {
      if (b < start || b > start + windowSamples) continue;
      ctx.beginPath();
      ctx.moveTo(xOf(b), PADDING.top);
      ctx.lineTo(xOf(b), PADDING.top + plotH);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    for (const s of series) {
      const visible = s.values.slice(start, start + windowSamples);
      const buckets = minMaxDecimate(visible, Math.max(1, Math.floor(plotW)));
      ctx.strokeStyle = resolveColor(canvas, s.color);
      ctx.globalAlpha = s.faded ? 0.45 : 1;
      ctx.lineWidth = s.faded ? 1 : 1.25;
      ctx.beginPath();
      buckets.forEach((b, i) => {
        const x = PADDING.left + (i / Math.max(1, buckets.length - 1)) * plotW;
        if (i === 0) ctx.moveTo(x, y(b.max));
        else ctx.lineTo(x, y(b.max));
        ctx.lineTo(x, y(b.min));
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [series, beats, width, height, start, windowSamples, sampleRate]);

  return (
    <figure className="chart">
      <figcaption className="chart-header">
        <span className="chart-title">{title}</span>
        <span className="chart-legend">
          {series.map((s) => (
            <span key={s.label} className="legend-item">
              <span className="legend-swatch" style={{ background: s.color, opacity: s.faded ? 0.5 : 1 }} />
              {s.label}
            </span>
          ))}
          {beats.length > 0 && (
            <span className="legend-item">
              <span className="legend-swatch legend-beat" />
              detected beats
            </span>
          )}
        </span>
      </figcaption>
      <div ref={wrapperRef} className="chart-canvas-wrap">
        <canvas ref={canvasRef} style={{ width: "100%", height }} role="img"
          aria-label={`${title}: ${series.map((s) => s.label).join(", ")}`} />
      </div>
      {maxOffset > 0 && (
        <label className="chart-scroll">
          <span>View from {(start / sampleRate).toFixed(1)} s</span>
          <input type="range" min={0} max={maxOffset} step={Math.round(sampleRate / 2)} value={start}
            onChange={(e) => setOffset(Number(e.target.value))} aria-label={`Scroll ${title}`} />
        </label>
      )}
    </figure>
  );
}
