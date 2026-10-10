"""Training losses."""
import torch
import torch.nn.functional as F

from .signal import envelope

STFT_RESOLUTIONS = ((64, 16), (256, 64), (1024, 256))  # (n_fft, hop) at 2 kHz


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x.squeeze(1).float(), n_fft, hop, window=window, return_complex=True)
    return spec.abs().clamp_min(1e-7)


def multi_resolution_stft_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Spectral convergence + log-magnitude L1 over several resolutions.

    Matches the frequency content of the output without requiring exact phase,
    which a plain L1 cannot do for noisy signals like heart sounds.
    """
    total = fake.new_zeros(())
    for n_fft, hop in STFT_RESOLUTIONS:
        f, r = _stft_mag(fake, n_fft, hop), _stft_mag(real, n_fft, hop)
        convergence = torch.linalg.norm(r - f) / torch.linalg.norm(r).clamp_min(1e-7)
        log_mag = F.l1_loss(f.log(), r.log())
        total = total + convergence + log_mag
    return total / len(STFT_RESOLUTIONS)


def envelope_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(envelope(fake), envelope(real))


def lsgan_d_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((real_score - 1) ** 2).mean() + (fake_score ** 2).mean())


def lsgan_g_loss(fake_score: torch.Tensor) -> torch.Tensor:
    return ((fake_score - 1) ** 2).mean()
