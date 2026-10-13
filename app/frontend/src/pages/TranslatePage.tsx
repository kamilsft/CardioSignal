import { useEffect, useState } from "react";
import { getSample, listSamples, translateFile, translateSignal } from "../api/client";
import type { Direction, SampleSummary, Translation } from "../api/schemas";
import { DirectionToggle } from "../components/DirectionToggle";
import { ResultView } from "../components/ResultView";
import { SamplePicker } from "../components/SamplePicker";
import { UploadForm } from "../components/UploadForm";
import { useAsyncTask } from "../hooks/useAsyncTask";
import { MODEL_CARD } from "../modelCard";

type Source = "sample" | "upload";

interface Outcome {
  translation: Translation;
  reference: number[] | null;
}

export function TranslatePage() {
  const [direction, setDirection] = useState<Direction>("ecg_to_pcg");
  const [source, setSource] = useState<Source>("sample");
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const { state, run, reset } = useAsyncTask<Outcome>();
  const busy = state.status === "loading";
  const inputName = direction === "ecg_to_pcg" ? "ECG" : "Heart sound";

  useEffect(() => {
    listSamples().then(setSamples).catch((e: Error) => setSamplesError(e.message));
  }, []);

  function translateSample(id: string, dir: Direction) {
    setSelectedSample(id);
    void run(async () => {
      const sample = await getSample(id);
      const [input, reference] = dir === "ecg_to_pcg" ? [sample.ecg, sample.pcg] : [sample.pcg, sample.ecg];
      return { translation: await translateSignal(input, sample.sample_rate, dir), reference };
    });
  }

  function changeDirection(dir: Direction) {
    setDirection(dir);
    if (source === "sample" && selectedSample) {
      translateSample(selectedSample, dir);
    } else {
      setSelectedSample(null);
      reset();
    }
  }

  function translateUpload(file: File, sampleRate: number | null) {
    setSelectedSample(null);
    void run(async () => ({ translation: await translateFile(file, direction, sampleRate), reference: null }));
  }

  return (
    <>
      <header className="hero">
        <p className="eyebrow">CycleGAN · Transformer · 1D U-Net</p>
        <h1>CardioSignal</h1>
        <p className="lead">
          Translate between the heart&apos;s electrical signal (ECG) and its sound (phonocardiogram).
          Pick a recording from a patient the model never saw, or upload your own.
        </p>
        <p className="badge" role="note">Research demo · not for diagnosis</p>
      </header>

      <main>
        <section className="panel" aria-labelledby="step-1">
          <h2 id="step-1"><span className="step">1</span> Choose a direction</h2>
          <DirectionToggle value={direction} onChange={changeDirection} disabled={busy} />
        </section>

        <section className="panel" aria-labelledby="step-2">
          <h2 id="step-2"><span className="step">2</span> Pick an input</h2>
          <div className="tabs" role="tablist">
            {(["sample", "upload"] as const).map((s) => (
              <button key={s} type="button" role="tab" aria-selected={source === s}
                className={source === s ? "tab active" : "tab"} onClick={() => setSource(s)}>
                {s === "sample" ? "Try a demo recording" : `Upload your ${direction === "ecg_to_pcg" ? "ECG" : "heart sound"}`}
              </button>
            ))}
          </div>
          {source === "sample" ? (
            samplesError ? <p className="form-error" role="alert">{samplesError}</p> : (
              <SamplePicker samples={samples} selectedId={selectedSample} disabled={busy}
                onSelect={(id) => translateSample(id, direction)} />
            )
          ) : (
            <UploadForm signalName={inputName} onSubmit={translateUpload} disabled={busy} />
          )}
        </section>

        <section className="panel" aria-labelledby="step-3">
          <h2 id="step-3"><span className="step">3</span> Result</h2>
          {state.status === "idle" && <p className="muted">Choose a demo recording or upload a file to see the translation.</p>}
          {state.status === "loading" && <p className="loading" role="status">Translating…</p>}
          {state.status === "error" && <p className="form-error" role="alert">{state.error}</p>}
          {state.status === "success" && <ResultView result={state.data.translation} reference={state.data.reference} />}
        </section>

        <section className="panel model-card" aria-labelledby="model">
          <h2 id="model">How well does it work?</h2>
          <p className="muted">Measured on {MODEL_CARD.testSegments.toLocaleString()} six-second segments from {MODEL_CARD.testPeople} people the model never saw · model {MODEL_CARD.version}</p>
          <div className="stats">
            {MODEL_CARD.metrics.map((m) => (
              <div key={m.detail} className="stat">
                <span className="stat-value">{m.value}</span>
                <span className="stat-label">{m.label}</span>
                <span className="stat-note">{m.detail}</span>
              </div>
            ))}
          </div>
          <ul className="limitations">
            {MODEL_CARD.limitations.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </section>
      </main>
    </>
  );
}
