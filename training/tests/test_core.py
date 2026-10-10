import numpy as np
import pytest
import torch

from cardiosignal.config import SEGMENT_LEN
from cardiosignal.data import recording_weights
from cardiosignal.inference import build_generator, selection_score, summarize
from cardiosignal.losses import envelope_loss, lsgan_d_loss, lsgan_g_loss, multi_resolution_stft_loss
from cardiosignal.metrics import envelope_pearson, mae, pearson
from cardiosignal.models import Discriminator
from cardiosignal.signal import downsample, fix_ecg_polarity, scale_to_unit


@pytest.fixture(scope="module")
def generator():
    torch.manual_seed(0)
    return build_generator().eval()


class TestModels:
    def test_generator_keeps_length_and_range(self, generator):
        x = torch.randn(2, 1, SEGMENT_LEN)
        with torch.no_grad():
            y = generator(x)
        assert y.shape == x.shape
        assert y.abs().max() <= 1.0

    def test_generator_handles_other_lengths(self, generator):
        with torch.no_grad():
            assert generator(torch.randn(1, 1, 2400)).shape == (1, 1, 2400)

    def test_discriminator_outputs_patches(self):
        d = Discriminator()
        out = d(torch.randn(2, 1, SEGMENT_LEN), torch.randn(2, 1, SEGMENT_LEN))
        assert out.shape[0] == 2 and out.shape[1] == 1 and out.shape[2] > 1

    def test_generator_gradients_flow(self, generator):
        g = build_generator().train()
        g(torch.randn(1, 1, 1200)).mean().backward()
        assert all(p.grad is not None for p in g.parameters() if p.requires_grad)


class TestSignal:
    def test_downsample_factor(self):
        assert downsample(np.random.randn(48000)).shape == (12000,)

    def test_scale_to_unit_range_and_immutability(self):
        x = np.array([1.0, 3.0, 5.0])
        y = scale_to_unit(x)
        assert np.allclose(y, [-1.0, 0.0, 1.0])
        assert np.allclose(x, [1.0, 3.0, 5.0])

    def test_scale_to_unit_flat_signal(self):
        assert np.all(scale_to_unit(np.ones(10)) == 0)

    @staticmethod
    def _synthetic_ecg():
        x = np.zeros(2000)
        x[100::400] = 1.0          # sharp upward R-peaks
        return x - 0.05 * np.sin(np.linspace(0, 20, 2000))

    def test_fix_ecg_polarity_flips_inverted(self):
        upright = self._synthetic_ecg()
        assert np.allclose(fix_ecg_polarity(-upright), upright)

    def test_fix_ecg_polarity_keeps_upright_and_input(self):
        upright = self._synthetic_ecg()
        before = upright.copy()
        assert np.allclose(fix_ecg_polarity(upright), upright)
        assert np.array_equal(upright, before)


class TestSampling:
    def test_recording_weights_power(self):
        recs = ["a"] * 4 + ["b"]
        w = recording_weights(recs, power=1.0)
        assert w[:4].sum() == pytest.approx(w[4:].sum())       # each recording equally likely
        w = recording_weights(recs, power=0.0)
        assert np.all(w == 1.0)                                 # natural frequencies
        w = recording_weights(recs, power=0.5)
        assert w[:4].sum() / w[4:].sum() == pytest.approx(2.0)  # sqrt(4) : sqrt(1)


class TestMetrics:
    def test_pearson_identical_and_inverted(self):
        x = torch.randn(3, 1, 500)
        assert torch.allclose(pearson(x, x), torch.ones(3, dtype=torch.float64))
        assert torch.allclose(pearson(x, -x), -torch.ones(3, dtype=torch.float64))

    def test_envelope_pearson_ignores_sign(self):
        x = torch.randn(2, 1, 2000)
        assert torch.allclose(envelope_pearson(x, -x), torch.ones(2, dtype=torch.float64))

    def test_mae_zero_for_identical(self):
        x = torch.randn(2, 1, 100)
        assert torch.all(mae(x, x) == 0)

    def test_summarize_with_mask_and_score(self):
        per_sample = {"ecg_to_pcg": {"envelope_pearson": [0.2, 0.6]}, "pcg_to_ecg": {"pearson": [0.4, 0.8]}}
        assert summarize(per_sample)["ecg_to_pcg"]["envelope_pearson"] == 0.4
        assert summarize(per_sample, mask=np.array([False, True]))["pcg_to_ecg"]["pearson"] == 0.8
        assert selection_score(summarize(per_sample)) == pytest.approx(1.0)


class TestLosses:
    def test_supervised_losses_zero_when_equal(self):
        x = torch.randn(2, 1, SEGMENT_LEN)
        assert envelope_loss(x, x) == 0
        assert multi_resolution_stft_loss(x, x) < 1e-5

    def test_stft_loss_positive_when_different(self):
        assert multi_resolution_stft_loss(torch.randn(1, 1, 4000), torch.randn(1, 1, 4000)) > 0.1

    def test_lsgan(self):
        ones, zeros = torch.ones(4), torch.zeros(4)
        assert lsgan_d_loss(ones, zeros) == 0
        assert lsgan_g_loss(ones) == 0
