"""Person-level train/val/test splitting.

Every recording is mapped to the *person* it came from, and whole people are
assigned to one split, so no one's heartbeats appear in both training and test.

The lab data mixes three cohorts:
  * cvd       normalized_Measure_<date>_C<person>S<session>_...  (heart disease)
  * measure   the same naming in the healthy folder               (healthy, several sessions each)
  * ephnogram normalized_ECGPCG<nnnn>_...                        (healthy; person from EPHNOGRAM metadata)
"""
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

MEASURE_PERSON = re.compile(r"_C(\d+)S\d+_")
EPHNOGRAM_RECORD = re.compile(r"(ECGPCG\d{4})")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Recording:
    path: str
    cohort: str      # "cvd", "measure" or "ephnogram"
    person: str      # globally unique person id, e.g. "cvd-C55", "ephnogram-S013"

    @property
    def is_cvd(self) -> bool:
        return self.cohort == "cvd"


def load_ephnogram_subjects(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())["record_to_subject"]


def identify(path: str | Path, is_cvd: bool, ephnogram_subjects: dict[str, str]) -> Recording:
    """Map one recording file to its cohort and person. Raises ValueError if the person is unknown."""
    name = Path(path).name
    measure = MEASURE_PERSON.search(name)
    if measure:
        cohort = "cvd" if is_cvd else "measure"
        return Recording(str(path), cohort, f"{cohort}-C{measure.group(1)}")
    record = EPHNOGRAM_RECORD.search(name)
    if record and not is_cvd:
        subject = ephnogram_subjects.get(record.group(1))
        if subject is None:
            raise ValueError(f"{name}: record {record.group(1)} is missing from the EPHNOGRAM subject list")
        return Recording(str(path), "ephnogram", f"ephnogram-{subject}")
    raise ValueError(f"Cannot tell which person {name} belongs to")


def split_people(recordings: list[Recording], counts: dict[str, tuple[int, int]], seed: int) -> dict[str, list[Recording]]:
    """Assign whole people to splits. counts[cohort] = (n_val_people, n_test_people); the rest train."""
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for cohort in sorted({r.cohort for r in recordings}):
        people = sorted({r.person for r in recordings if r.cohort == cohort})
        rng.shuffle(people)
        n_val, n_test = counts.get(cohort, (0, 0))
        if n_val + n_test >= len(people):
            raise ValueError(f"Cohort {cohort} has {len(people)} people; cannot hold out {n_val} + {n_test}")
        for i, person in enumerate(people):
            assignment[person] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return {split: [r for r in recordings if assignment[r.person] == split] for split in SPLITS}


def assert_no_person_overlap(splits: dict[str, list[Recording]]) -> None:
    people = {split: {r.person for r in recs} for split, recs in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and people[a] & people[b]:
                raise AssertionError(f"People in both {a} and {b}: {sorted(people[a] & people[b])}")
