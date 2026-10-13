import io

import numpy as np
import pytest
from scipy.io import wavfile

from cardiosignal_api.signal_io import SignalError, parse_signal_file, validate_signal


def _wav_bytes(x, rate):
    buf = io.BytesIO()
    wavfile.write(buf, rate, x.astype(np.int16))
    return buf.getvalue()


class TestParse:
    def test_csv_with_header_and_extra_columns(self):
        x, rate = parse_signal_file("ecg.csv", b"value,time\n0.1,0\n0.2,1\n-0.3,2\n", 500)
        assert x.tolist() == [0.1, 0.2, -0.3] and rate == 500

    def test_txt_whitespace(self):
        x, _ = parse_signal_file("s.txt", b"1 9\n2 9\n\n3 9\n", 1000)
        assert x.tolist() == [1.0, 2.0, 3.0]

    def test_text_rejects_garbage_after_data(self):
        with pytest.raises(SignalError, match="line 3"):
            parse_signal_file("s.csv", b"1\n2\nabc\n", 1000)

    def test_npy_roundtrip_and_no_pickle(self):
        buf = io.BytesIO()
        np.save(buf, np.arange(5.0))
        x, _ = parse_signal_file("s.npy", buf.getvalue(), 100)
        assert x.tolist() == [0, 1, 2, 3, 4]
        pickled = io.BytesIO()
        np.save(pickled, np.array([{"a": 1}], dtype=object), allow_pickle=True)
        with pytest.raises(SignalError, match="pickled"):
            parse_signal_file("s.npy", pickled.getvalue(), 100)

    def test_wav_uses_its_own_rate_and_first_channel(self):
        stereo = np.stack([np.arange(10), -np.arange(10)], axis=1)
        x, rate = parse_signal_file("s.wav", _wav_bytes(stereo, 4000), None)
        assert rate == 4000 and x.tolist() == list(range(10))

    def test_requires_rate_for_csv(self):
        with pytest.raises(SignalError, match="sample rate"):
            parse_signal_file("s.csv", b"1\n2\n", None)

    def test_npy_with_text_values_is_a_friendly_error(self):
        buf = io.BytesIO()
        np.save(buf, np.array(["abc"] * 5000))
        signal, rate = parse_signal_file("s.npy", buf.getvalue(), 1000)
        with pytest.raises(SignalError, match="numbers only"):
            validate_signal(signal, rate, 3, 60)

    def test_corrupt_wav(self):
        with pytest.raises(SignalError, match="WAV"):
            parse_signal_file("s.wav", b"RIFF\x00\x00garbage", None)

    def test_rejects_unknown_extension(self):
        with pytest.raises(SignalError, match="Unsupported"):
            parse_signal_file("s.exe", b"", 100)


class TestValidate:
    def test_accepts_and_flattens_column(self):
        x = validate_signal(np.random.randn(4000, 1), 1000, 3, 60)
        assert x.shape == (4000,)

    @pytest.mark.parametrize("signal, rate, message", [
        (np.random.randn(1000), 1000, "at least"),
        (np.random.randn(70_000), 1000, "maximum"),
        (np.ones(5000), 1000, "flat"),
        (np.array([1.0, np.nan] * 3000), 1000, "NaN"),
        (np.random.randn(5000), 100, "Sample rate"),
        (np.random.randn(200_000), 47999, "Unsupported sample rate"),
        (np.random.randn(10, 10), 1000, "single-channel"),
    ])
    def test_rejects_bad_input(self, signal, rate, message):
        with pytest.raises(SignalError, match=message):
            validate_signal(signal, rate, 3, 60)
