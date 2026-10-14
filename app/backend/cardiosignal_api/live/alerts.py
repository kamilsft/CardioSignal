"""Alert rules on top of per-second vitals: open after a condition persists, resolve after it clears.

Both directions need the condition to hold for a while (hysteresis), so one noisy reading
neither raises an alarm nor silences a real one.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable, Literal

from .monitor import Vitals

Severity = Literal["critical", "warning"]
HIGH_HR_BPM = 150          # critical
LOW_HR_BPM = 40            # critical
FAST_HR_BPM = 120          # warning, up to the critical limit
SLOW_HR_BPM = 50           # warning, down to the critical limit
IRREGULARITY_THRESHOLD = 0.15     # replayed sinus-rhythm recordings measure 0.01-0.06


@dataclass(frozen=True)
class AlertRule:
    kind: str
    severity: Severity
    title: str
    onset_seconds: float
    clear_seconds: float
    triggered: Callable[[Vitals], bool]


def _hr_above(limit: float) -> Callable[[Vitals], bool]:
    return lambda v: v.heart_rate_bpm is not None and v.heart_rate_bpm > limit


def _hr_below(limit: float) -> Callable[[Vitals], bool]:
    return lambda v: v.heart_rate_bpm is not None and v.heart_rate_bpm < limit


def _hr_between(low: float, high: float) -> Callable[[Vitals], bool]:
    """low <= hr <= high, so a warning band never overlaps its critical rule."""
    return lambda v: v.heart_rate_bpm is not None and low <= v.heart_rate_bpm <= high


RULES: tuple[AlertRule, ...] = (
    AlertRule("high_heart_rate", "critical", f"Heart rate above {HIGH_HR_BPM} bpm", 10, 5, _hr_above(HIGH_HR_BPM)),
    AlertRule("low_heart_rate", "critical", f"Heart rate below {LOW_HR_BPM} bpm", 10, 5, _hr_below(LOW_HR_BPM)),
    AlertRule("fast_heart_rate", "warning", f"Heart rate above {FAST_HR_BPM} bpm", 15, 10,
              _hr_between(FAST_HR_BPM + 1e-9, HIGH_HR_BPM)),
    AlertRule("slow_heart_rate", "warning", f"Heart rate below {SLOW_HR_BPM} bpm", 15, 10,
              _hr_between(LOW_HR_BPM, SLOW_HR_BPM - 1e-9)),
    AlertRule("irregular_rhythm", "warning", "Irregular rhythm", 15, 10,
              lambda v: v.rr_irregularity is not None and v.rr_irregularity > IRREGULARITY_THRESHOLD),
    AlertRule("signal_lost", "warning", "No ECG signal", 5, 3, lambda v: v.status == "no_signal"),
)
SEVERITY_ORDER = {"critical": 0, "warning": 1}


@dataclass(frozen=True)
class Alert:
    id: str
    patient_id: str
    kind: str
    severity: Severity
    title: str
    started_at: float
    resolved_at: float | None = None
    heart_rate_bpm: float | None = None      # at onset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Track:
    """Per patient and rule. Internal, mutable: updated every tick for every patient."""
    pending_since: float | None = None
    clear_since: float | None = None
    alert: Alert | None = None


class AlertEngine:
    def __init__(self, rules: tuple[AlertRule, ...] = RULES, history: int = 50):
        self.rules = rules
        self.history = history
        self._tracks: dict[tuple[str, str], _Track] = {}
        self._recent: OrderedDict[str, Alert] = OrderedDict()

    def update(self, vitals: Iterable[Vitals], now: float) -> list[Alert]:
        """Apply one tick of vitals; returns alerts that opened or resolved."""
        changes = []
        for v in vitals:
            for rule in self.rules:
                track = self._tracks.setdefault((v.patient_id, rule.kind), _Track())
                change = self._step(track, rule, v, now)
                if change is not None:
                    changes.append(change)
        return changes

    def forget(self, keep: set[str], now: float) -> list[Alert]:
        """Drop patients not in `keep`, resolving their open alerts."""
        changes = []
        for key in [k for k in self._tracks if k[0] not in keep]:
            track = self._tracks.pop(key)
            if track.alert is not None:
                changes.append(self._resolve(track.alert, now))
        return changes

    def active(self) -> list[Alert]:
        alerts = [t.alert for t in self._tracks.values() if t.alert is not None]
        return sorted(alerts, key=lambda a: (SEVERITY_ORDER[a.severity], a.started_at, a.patient_id))

    def active_kinds(self, patient_id: str) -> list[str]:
        return [r.kind for r in self.rules
                if (t := self._tracks.get((patient_id, r.kind))) is not None and t.alert is not None]

    def recent(self) -> list[Alert]:
        """Latest alerts, newest first, open or resolved."""
        return list(reversed(self._recent.values()))

    def _step(self, track: _Track, rule: AlertRule, v: Vitals, now: float) -> Alert | None:
        triggered = rule.triggered(v)
        if track.alert is None:
            if not triggered:
                track.pending_since = None
                return None
            track.pending_since = now if track.pending_since is None else track.pending_since
            if now - track.pending_since < rule.onset_seconds:
                return None
            track.alert = Alert(f"{v.patient_id}:{rule.kind}:{int(now * 1000)}", v.patient_id, rule.kind,
                                rule.severity, rule.title, now, heart_rate_bpm=v.heart_rate_bpm)
            track.pending_since = None
            self._remember(track.alert)
            return track.alert
        if triggered:
            track.clear_since = None
            return None
        track.clear_since = now if track.clear_since is None else track.clear_since
        if now - track.clear_since < rule.clear_seconds:
            return None
        resolved = self._resolve(track.alert, now)
        track.alert, track.clear_since = None, None
        return resolved

    def _resolve(self, alert: Alert, now: float) -> Alert:
        resolved = replace(alert, resolved_at=now)
        if alert.id in self._recent:
            self._recent[alert.id] = resolved
        return resolved

    def _remember(self, alert: Alert) -> None:
        self._recent[alert.id] = alert
        while len(self._recent) > self.history:
            self._recent.popitem(last=False)
