import numpy as np

from baseline_beats import heart_sound_beats

FS = 2000


def test_finds_loud_sounds_and_applies_offset():
    pcg = np.zeros(FS * 6)
    sounds = np.arange(1000, len(pcg), 1600)          # one loud "lub" every 0.8 s
    for s in sounds:
        pcg[s:s + 60] = np.sin(np.linspace(0, 20 * np.pi, 60))
    beats = heart_sound_beats(pcg, smooth_ms=40, min_gap_s=0.4, height=0.3, offset_ms=-50)
    assert len(beats) == len(sounds)
    assert np.all(np.abs(beats - (sounds + 30 - 100)) < 30)   # sound centre, shifted 50 ms earlier


def test_silence_has_no_beats():
    assert len(heart_sound_beats(np.zeros(FS * 6), 40, 0.4, 0.3, -50)) == 0
