import type { Translation } from "../api/schemas";
import { formatBpm } from "../lib/signal";
import { PlayButton } from "./PlayButton";
import { SignalChart, type Series } from "./SignalChart";

interface Props {
  result: Translation;
  /** The real recording of the output modality, when translating a demo sample. */
  reference?: readonly number[] | null;
}

const ECG_COLOR = "var(--ecg)";
const PCG_COLOR = "var(--pcg)";
const REFERENCE_COLOR = "var(--reference)";

function StatTile({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: "warn" | "ok" }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

export function ResultView({ result, reference }: Props) {
  const toPcg = result.direction === "ecg_to_pcg";
  const ecgAnalysis = toPcg ? result.input_analysis : result.output_analysis;
  const pcgAnalysis = toPcg ? result.output_analysis : result.input_analysis;
  const outputName = toPcg ? "Generated heart sound" : "Generated ECG";
  const outputSeries: Series[] = [
    ...(reference ? [{ label: toPcg ? "real heart sound" : "real ECG", values: reference, color: REFERENCE_COLOR, faded: true }] : []),
    { label: toPcg ? "generated heart sound" : "generated ECG", values: result.output_signal, color: toPcg ? PCG_COLOR : ECG_COLOR },
  ];
  const realPcg = toPcg ? reference : result.input_signal;

  return (
    <section className="results" aria-live="polite">
      {result.warnings.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}

      <div className="stats">
        <StatTile label="Heart rate (input)" value={formatBpm(result.input_analysis.heart_rate_bpm)} />
        <StatTile label="Heart rate (generated)" value={formatBpm(result.output_analysis.heart_rate_bpm)} />
        <StatTile label="ECG beats detected" value={String(ecgAnalysis.beats.length)}
          note={toPcg ? "in the input ECG" : "in the generated ECG"} />
        {!toPcg && pcgAnalysis.quality != null && (
          <StatTile label="Heart-sound quality" value={pcgAnalysis.quality_ok ? "Good" : "Low"}
            note={`rhythm score ${pcgAnalysis.quality.toFixed(2)} · ${pcgAnalysis.quality_ok ? "clear rhythm" : "weak rhythm: low confidence"}`}
            tone={pcgAnalysis.quality_ok ? "ok" : "warn"} />
        )}
      </div>

      <SignalChart title={toPcg ? "Input ECG" : "Input heart sound"} sampleRate={result.sample_rate}
        series={[{ label: toPcg ? "ECG" : "heart sound", values: result.input_signal, color: toPcg ? ECG_COLOR : PCG_COLOR }]}
        beats={toPcg ? result.input_analysis.beats : []} />
      <SignalChart title={outputName} sampleRate={result.sample_rate} series={outputSeries}
        beats={toPcg ? [] : result.output_analysis.beats} />

      <div className="audio-row">
        {toPcg && <PlayButton signal={result.output_signal} sampleRate={result.sample_rate} label="Play generated heart sound" />}
        {realPcg && <PlayButton signal={realPcg} sampleRate={result.sample_rate} label="Play real heart sound" />}
        <small className="muted">Headphones recommended: heart sounds are low-pitched.</small>
      </div>
    </section>
  );
}
