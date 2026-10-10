"""Generator (1D U-Net with Transformer bottleneck) and conditional patch discriminator."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(8, channels)


class DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
            _norm(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, 5, padding=2),
            _norm(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample to the skip connection's length, concatenate, then convolve."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in + c_skip, c_out, 5, padding=2),
            _norm(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, 5, padding=2),
            _norm(c_out),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.net(torch.cat([x, skip], dim=1))


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal encoding so attention knows where each token sits in time."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: [B, T, D]
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class Generator(nn.Module):
    """[B, 1, T] -> [B, 1, T] in [-1, 1]. Works for any T divisible by prod(strides)."""

    def __init__(self, channels=(64, 128, 256), strides=(4, 3, 2),
                 heads: int = 4, layers: int = 4, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3 = channels
        s1, s2, s3 = strides
        self.stem = nn.Sequential(nn.Conv1d(1, c1 // 2, 7, padding=3), _norm(c1 // 2), nn.GELU())
        self.down1 = DownBlock(c1 // 2, c1, 9, s1)
        self.down2 = DownBlock(c1, c2, 7, s2)
        self.down3 = DownBlock(c2, c3, 5, s3)

        self.pos = SinusoidalPositionalEncoding(c3)
        layer = nn.TransformerEncoderLayer(c3, heads, ff_dim, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.bottleneck_norm = nn.LayerNorm(c3)

        self.up3 = UpBlock(c3, c2, c2)
        self.up2 = UpBlock(c2, c1, c1)
        self.up1 = UpBlock(c1, c1 // 2, c1 // 2)
        self.head = nn.Conv1d(c1 // 2, 1, 7, padding=3)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        t = self.pos(x3.transpose(1, 2))
        t = self.bottleneck_norm(self.transformer(t)).transpose(1, 2)

        y = self.up3(t + x3, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)
        return torch.tanh(self.head(y))


class Discriminator(nn.Module):
    """Conditional PatchGAN: scores (condition, signal) pairs patch by patch.

    Conditioning on the source signal means the discriminator judges whether the
    output *matches its input*, not only whether it looks realistic.
    No sigmoid: trained with least-squares GAN loss.
    """

    def __init__(self, in_channels: int = 2, base: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(in_channels, base, 15, stride=4, padding=7)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(base, base * 2, 15, stride=4, padding=7)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(base * 2, base * 4, 15, stride=4, padding=7)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(base * 4, base * 4, 7, padding=3)), nn.LeakyReLU(0.2),
            nn.Conv1d(base * 4, 1, 7, padding=3),
        )

    def forward(self, condition, signal):
        return self.net(torch.cat([condition, signal], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
