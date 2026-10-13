import numpy as np

from beat_accuracy import beat_scores, per_group_f1


def _row(recording, person, real, fake):
    return {"recording": recording, "person": person, "real_peaks": np.array(real), "fake_peaks": np.array(fake)}


def test_beat_scores_precision_recall():
    s = beat_scores([(np.array([100, 1000]), np.array([110, 2000]))], tolerance=50)
    assert s["recall_pct"] == 50.0 and s["precision_pct"] == 50.0 and s["f1_pct"] == 50.0


def test_per_person_groups_recordings_of_the_same_person():
    rows = [_row("rec1", "p1", [100], [100]), _row("rec2", "p1", [100], [900]),   # p1: 1 of 2 beats right
            _row("rec3", "p2", [100], [100])]                                      # p2: perfect
    by_person = per_group_f1(rows, 50, "person")
    by_recording = per_group_f1(rows, 50, "recording")
    assert by_person["count"] == 2 and by_recording["count"] == 3
    assert by_person["per_person_f1_pct"] == {"p1": 50.0, "p2": 100.0}
    assert by_person["median_f1_pct"] == 75.0 and by_person["at_least_80_pct"] == 1
