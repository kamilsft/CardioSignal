import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=4, padding=7),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, kernel_size=15, stride=4, padding=7),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x) 

if __name__ == "__main__":
    model = Discriminator()
    sample = torch.randn(4, 1, 48000)  # (batch, channels, sequence length)
    output = model(sample)
    print("Discriminator output shape:", output.shape)
