import type { LiveSummary, PatientVitals } from "../api/liveSchemas";
import { useLiveStream, type Connection } from "../hooks/useLiveStream";
import { PatientTile, STATUS_LABEL } from "./PatientTile";

const WINDOW_SECONDS = 6;
const MAX_LISTED = 500;
const SIMULATOR_COMMAND = "LIVE_INGEST_TOKEN=<token> python -m cardiosignal_api.live.simulator --patients 50";

const CONNECTION_LABEL: Record<Connection, string> = {
  connecting: "Connecting…",
  live: "Live",
  reconnecting: "Reconnecting…",
};

function SummaryStats({ summary }: { summary: LiveSummary }) {
  const items = [
    { label: "Patients", value: summary.patients.toLocaleString() },
    { label: "Receiving signal", value: summary.ok.toLocaleString(), tone: "stat-ok" },
    { label: "Connecting", value: summary.warming_up.toLocaleString() },
    { label: "No signal", value: summary.no_signal.toLocaleString(), tone: summary.no_signal > 0 ? "stat-warn" : "" },
    { label: "Analysis time per tick", value: `${Math.round(summary.analysis_ms)} ms` },
  ];
  return (
    <section className="stats live-stats" aria-label="Summary">
      {items.map((item) => (
        <div key={item.label} className={`stat ${item.tone ?? ""}`}>
          <span className="stat-value">{item.value}</span>
          <span className="stat-label">{item.label}</span>
        </div>
      ))}
    </section>
  );
}

function PatientList({ patients }: { patients: PatientVitals[] }) {
  const shown = patients.slice(0, MAX_LISTED);
  return (
    <>
      <ul className="patient-list">
        {shown.map((p) => (
          <li key={p.id} className={`patient-chip status-${p.status}`}>
            <span className="patient-id">{p.id}</span>
            <span>{p.heart_rate_bpm !== null ? `${Math.round(p.heart_rate_bpm)} bpm` : STATUS_LABEL[p.status]}</span>
          </li>
        ))}
      </ul>
      {patients.length > shown.length && (
        <p className="muted">and {(patients.length - shown.length).toLocaleString()} more</p>
      )}
    </>
  );
}

export function LiveMonitor() {
  const { connection, summary, patients, waveforms } = useLiveStream(WINDOW_SECONDS);
  const watched = patients.filter((p) => waveforms[p.id]);
  const others = patients.filter((p) => !waveforms[p.id]);

  return (
    <>
      <header className="hero">
        <p className="eyebrow">Real-time · WebSockets</p>
        <h1>Live monitor</h1>
        <p className="lead">
          ECG streams from bedside devices, analysed every second for heart rate. The patients here are simulated:
          real recordings replayed at real-time speed.
        </p>
        <p className={`connection connection-${connection}`} role="status">
          <span className="connection-dot" aria-hidden="true" />
          {CONNECTION_LABEL[connection]}
        </p>
      </header>

      <main>
        {summary && <SummaryStats summary={summary} />}

        {summary === null && <section className="panel"><p className="muted">Waiting for data…</p></section>}

        {summary !== null && summary.patients === 0 && (
          <section className="panel">
            <p className="muted">No patients are streaming yet. Start the simulator from <code>app/backend</code>:</p>
            <pre className="command"><code>{SIMULATOR_COMMAND}</code></pre>
          </section>
        )}

        {watched.length > 0 && (
          <section className="panel" aria-labelledby="live-waveforms">
            <h2 id="live-waveforms">Live ECG <span className="muted">· last {WINDOW_SECONDS} s</span></h2>
            <div className="patient-grid">
              {watched.map((p) => (
                <PatientTile key={p.id} patient={p} waveform={waveforms[p.id]} windowSeconds={WINDOW_SECONDS} />
              ))}
            </div>
          </section>
        )}

        {others.length > 0 && (
          <section className="panel" aria-labelledby="live-others">
            <h2 id="live-others">All other patients</h2>
            <PatientList patients={others} />
          </section>
        )}
      </main>
    </>
  );
}
