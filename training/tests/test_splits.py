import pytest

from splits import Recording, assert_no_person_overlap, identify, split_people

EPH = {"ECGPCG0001": "S023", "ECGPCG0002": "S023", "ECGPCG0003": "S003"}


def test_identify_each_cohort():
    assert identify("x/normalized_Measure_20100405_C40S10_HDF_segments.pt", True, EPH) == Recording(
        "x/normalized_Measure_20100405_C40S10_HDF_segments.pt", "cvd", "cvd-C40")
    assert identify("normalized_Measure_20100423_C46S10_HDF_segments.pt", False, EPH).person == "measure-C46"
    assert identify("normalized_ECGPCG0002_segments.pt", False, EPH).person == "ephnogram-S023"


def test_identify_rejects_unknown_people():
    with pytest.raises(ValueError, match="missing"):
        identify("normalized_ECGPCG0099_segments.pt", False, EPH)
    with pytest.raises(ValueError, match="Cannot tell"):
        identify("mystery.pt", False, EPH)


def _recordings():
    recs = []
    for person in range(10):                       # several sessions per person, like the real data
        for session in range(3):
            recs.append(identify(f"normalized_Measure_2010_C{person}S{session}_HDF.pt", False, EPH))
            recs.append(identify(f"normalized_Measure_2010_C{100 + person}S{session}_HDF.pt", True, EPH))
    return recs


def test_split_keeps_every_person_in_one_split():
    splits = split_people(_recordings(), {"measure": (2, 3), "cvd": (1, 2)}, seed=0)
    assert_no_person_overlap(splits)
    assert len({r.person for r in splits["test"] if r.cohort == "measure"}) == 3
    assert len({r.person for r in splits["val"] if r.cohort == "cvd"}) == 1
    assert sum(len(v) for v in splits.values()) == len(_recordings())


def test_split_is_reproducible():
    a = split_people(_recordings(), {"measure": (2, 3)}, seed=7)
    b = split_people(_recordings(), {"measure": (2, 3)}, seed=7)
    assert [r.path for r in a["test"]] == [r.path for r in b["test"]]


def test_overlap_detector():
    r = identify("normalized_Measure_2010_C1S1_HDF.pt", False, EPH)
    with pytest.raises(AssertionError, match="measure-C1"):
        assert_no_person_overlap({"train": [r], "val": [], "test": [r]})


def test_cannot_hold_out_everyone():
    with pytest.raises(ValueError, match="cannot hold out"):
        split_people(_recordings(), {"measure": (5, 5)}, seed=0)
