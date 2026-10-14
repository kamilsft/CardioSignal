import pytest

from cardiosignal_api.live.alerts import RULES, AlertEngine
from cardiosignal_api.live.monitor import Vitals


def ok(pid="p1", bpm=72.0, irregularity=0.03):
    return Vitals(pid, bpm, "ok", irregularity)


def lost(pid="p1"):
    return Vitals(pid, None, "no_signal")


def run(engine, vitals_by_second):
    """Feed one tick per second; returns every alert change."""
    changes = []
    for t, vitals in enumerate(vitals_by_second):
        changes += engine.update(vitals if isinstance(vitals, list) else [vitals], now=float(t))
    return changes


def test_sustained_high_heart_rate_opens_one_alert():
    engine = AlertEngine()
    changes = run(engine, [ok(bpm=165)] * 15)
    assert [(a.kind, a.resolved_at) for a in changes] == [("high_heart_rate", None)]
    alert = changes[0]
    assert alert.started_at == 10.0 and alert.severity == "critical" and alert.heart_rate_bpm == 165
    assert [a.kind for a in engine.active()] == ["high_heart_rate"]


def test_short_spike_does_not_alert():
    engine = AlertEngine()
    assert run(engine, [ok(bpm=165)] * 5 + [ok()] * 10 + [ok(bpm=165)] * 5) == []


def test_alert_resolves_only_after_staying_normal():
    engine = AlertEngine()
    run(engine, [ok(bpm=165)] * 11)
    # one normal tick in the middle of the episode must not resolve it (hysteresis)
    assert engine.update([ok()], now=11.0) == []
    assert engine.update([ok(bpm=165)], now=12.0) == []
    changes = run_from(engine, 13, [ok()] * 6)
    assert len(changes) == 1 and changes[0].resolved_at == 18.0
    assert engine.active() == []


def run_from(engine, start, vitals_list):
    changes = []
    for i, v in enumerate(vitals_list):
        changes += engine.update([v], now=float(start + i))
    return changes


@pytest.mark.parametrize("vitals, kind", [
    (ok(bpm=33), "low_heart_rate"),
    (ok(irregularity=0.3), "irregular_rhythm"),
    (lost(), "signal_lost"),
])
def test_each_rule_fires(vitals, kind):
    engine = AlertEngine()
    run(engine, [vitals] * 20)
    assert [a.kind for a in engine.active()] == [kind]


def test_patients_are_independent():
    engine = AlertEngine()
    run(engine, [[ok("a", bpm=165), ok("b")]] * 12)
    assert [a.patient_id for a in engine.active()] == ["a"]
    assert engine.active_kinds("a") == ["high_heart_rate"] and engine.active_kinds("b") == []


def test_forget_resolves_alerts_of_removed_patients():
    engine = AlertEngine()
    run(engine, [lost("gone")] * 8)
    changes = engine.forget(keep=set(), now=9.0)
    assert [(a.patient_id, a.resolved_at) for a in changes] == [("gone", 9.0)]
    assert engine.active() == []


def test_recent_lists_newest_first_and_is_bounded():
    engine = AlertEngine(history=2)
    for i, pid in enumerate(["a", "b", "c"]):
        for t in range(6):
            engine.update([lost(pid)], now=i * 100 + t)
    assert [a.patient_id for a in engine.recent()] == ["c", "b"]


def test_resolved_alert_is_updated_in_recent():
    engine = AlertEngine()
    run(engine, [lost()] * 6)
    run_from(engine, 6, [ok()] * 5)
    [alert] = engine.recent()
    assert alert.resolved_at is not None


def test_critical_alerts_are_listed_first():
    engine = AlertEngine()
    run(engine, [[lost("a"), ok("b", bpm=170)]] * 12)
    assert [a.severity for a in engine.active()] == ["critical", "warning"]


def test_alert_serialises_to_json_friendly_dict():
    engine = AlertEngine()
    [alert] = run(engine, [ok(bpm=165)] * 11)
    data = alert.to_dict()
    assert data["id"] == "p1:high_heart_rate:10000" and data["title"] == RULES[0].title
    assert set(data) == {"id", "patient_id", "kind", "severity", "title", "started_at", "resolved_at",
                         "heart_rate_bpm"}
