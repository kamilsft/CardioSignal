import type { SampleSummary } from "../api/schemas";

interface Props {
  samples: SampleSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  disabled?: boolean;
}

export function SamplePicker({ samples, selectedId, onSelect, disabled }: Props) {
  if (samples.length === 0) {
    return <p className="muted">No demo recordings available.</p>;
  }
  return (
    <ul className="sample-grid" aria-label="Demo recordings">
      {samples.map((s) => (
        <li key={s.id}>
          <button type="button" className={s.id === selectedId ? "sample-card selected" : "sample-card"}
            onClick={() => onSelect(s.id)} disabled={disabled} aria-pressed={s.id === selectedId}>
            <span className={`tag tag-${s.label}`}>{s.label === "cvd" ? "Heart disease" : "Healthy"}</span>
            <span className="sample-name">{s.description}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
