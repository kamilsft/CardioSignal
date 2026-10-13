import type { Direction } from "../api/schemas";

const OPTIONS: { value: Direction; label: string; hint: string }[] = [
  { value: "ecg_to_pcg", label: "ECG → Heart sound", hint: "Generate a phonocardiogram from an ECG" },
  { value: "pcg_to_ecg", label: "Heart sound → ECG", hint: "Reconstruct an ECG from a phonocardiogram" },
];

interface Props {
  value: Direction;
  onChange: (direction: Direction) => void;
  disabled?: boolean;
}

export function DirectionToggle({ value, onChange, disabled }: Props) {
  return (
    <div className="segmented" role="radiogroup" aria-label="Translation direction">
      {OPTIONS.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} title={o.hint}
          className={value === o.value ? "segmented-option active" : "segmented-option"}
          onClick={() => onChange(o.value)} disabled={disabled}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
