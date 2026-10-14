"""Analysis for one shard of patients: buffers in, vitals + alerts + waveform chunks out.

Used in-process by the memory hub and by each Redis worker. A tick has three phases so the
CPU-heavy part can leave the event loop without racing incoming samples:

    collect()   on the event loop: drop stale patients, copy each buffer
    compute()   in a thread: QRS detection on the copies (pure function)
    finalize()  on the event loop: alert rules, JSON-ready result
"""
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .alerts import SEVERITY_ORDER, AlertEngine
from .monitor import LIVE_FS, PatientStream, Vitals, vitals_for

WAVEFORM_DECIMATION = 2       # 250 Hz -> 125 Hz is plenty for a small tile and halves traffic
WAVEFORM_DECIMALS = 3
RECENT_ALERTS = 50


@dataclass(frozen=True)
class Reading:
    patient_id: str
    window: np.ndarray
    last_seen: float | None
    new_samples: np.ndarray | None    # only for patients whose waveform is sent


def compute_vitals(readings: list[Reading], now: float, fs: int = LIVE_FS) -> tuple[list[Vitals], float]:
    started = time.perf_counter()
    vitals = [vitals_for(r.patient_id, r.window, r.last_seen, now, fs) for r in readings]
    return vitals, (time.perf_counter() - started) * 1000


def _decimate(x: np.ndarray) -> list[float]:
    averaged = x.reshape(-1, WAVEFORM_DECIMATION).mean(axis=1)     # take_new() returns whole pairs
    return np.round(averaged.astype(np.float64), WAVEFORM_DECIMALS).tolist()


class ShardAnalyzer:
    STALE_SECONDS = 30        # a patient with no data this long is removed (shown as "no signal" until then)

    def __init__(self, fs: int = LIVE_FS, max_waveforms: int = 24, alerts: AlertEngine | None = None):
        self.fs = fs
        self.max_waveforms = max_waveforms
        self.alerts = alerts or AlertEngine(history=RECENT_ALERTS)
        self._streams: dict[str, PatientStream] = {}

    def __len__(self) -> int:
        return len(self._streams)

    def __contains__(self, patient_id: str) -> bool:
        return patient_id in self._streams

    def add(self, patient_id: str) -> None:
        self._streams.setdefault(patient_id, PatientStream(patient_id, self.fs))

    def ingest(self, patient_id: str, samples: np.ndarray, now: float) -> None:
        self.add(patient_id)
        self._streams[patient_id].push(samples, now)

    def collect(self, now: float, keep: frozenset[str] = frozenset()) -> list[Reading]:
        """Copy every buffer. Patients in `keep` (connected devices) are never dropped as stale."""
        for pid in [p for p, s in self._streams.items() if p not in keep
                    and (s.last_seen is None or now - s.last_seen > self.STALE_SECONDS)]:
            del self._streams[pid]
        # Alerting patients first, so their waveforms are the ones sent when there are too many to show.
        order = sorted(self._streams, key=lambda pid: (not self.alerts.active_kinds(pid), pid))
        readings = []
        for index, pid in enumerate(order):
            stream = self._streams[pid]
            new = stream.take_new(WAVEFORM_DECIMATION) if index < self.max_waveforms else None
            readings.append(Reading(pid, stream.window(), stream.last_seen, new))
        return readings

    def finalize(self, readings: list[Reading], vitals: list[Vitals], analysis_ms: float,
                 now: float) -> dict[str, Any]:
        self.alerts.forget(keep={v.patient_id for v in vitals}, now=now)
        self.alerts.update(vitals, now)
        patients = [{"id": v.patient_id, "heart_rate_bpm": v.heart_rate_bpm, "status": v.status,
                     "rr_irregularity": v.rr_irregularity, "alerts": self.alerts.active_kinds(v.patient_id)}
                    for v in vitals]
        return {
            "time": round(now, 3),
            "analysis_ms": round(analysis_ms, 1),
            "patients": patients,
            "waveforms": {r.patient_id: {"fs": self.fs // WAVEFORM_DECIMATION, "samples": _decimate(r.new_samples)}
                          for r in readings if r.new_samples is not None},
            "alerts": {"active": [a.to_dict() for a in self.alerts.active()],
                       "recent": [a.to_dict() for a in self.alerts.recent()]},
        }

    def tick(self, now: float, keep: frozenset[str] = frozenset()) -> dict[str, Any]:
        """All three phases synchronously (tests and simple callers)."""
        readings = self.collect(now, keep)
        vitals, analysis_ms = compute_vitals(readings, now, self.fs)
        return self.finalize(readings, vitals, analysis_ms, now)


def merge_results(results: list[dict[str, Any]], max_waveforms: int, now: float,
                  extra_summary: dict[str, Any]) -> dict[str, Any]:
    """Combine shard results into the one snapshot dashboards receive."""
    patients = sorted((p for r in results for p in r["patients"]), key=lambda p: (not p["alerts"], p["id"]))
    all_waveforms = {pid: w for r in results for pid, w in r["waveforms"].items()}
    waveforms = {}
    for p in patients:
        if p["id"] in all_waveforms and len(waveforms) < max_waveforms:
            waveforms[p["id"]] = all_waveforms[p["id"]]
    active = sorted((a for r in results for a in r["alerts"]["active"]),
                    key=lambda a: (SEVERITY_ORDER[a["severity"]], a["started_at"], a["patient_id"]))
    recent = sorted((a for r in results for a in r["alerts"]["recent"]),
                    key=lambda a: a["started_at"], reverse=True)[:RECENT_ALERTS]
    counts = {status: sum(p["status"] == status for p in patients) for status in ("ok", "warming_up", "no_signal")}
    return {
        "type": "snapshot",
        "time": round(now, 3),
        "summary": {"patients": len(patients), **counts, "alerts_active": len(active),
                    "analysis_ms": max((r["analysis_ms"] for r in results), default=0.0), **extra_summary},
        "patients": patients,
        "waveforms": waveforms,
        "alerts": {"active": active, "recent": recent},
    }
