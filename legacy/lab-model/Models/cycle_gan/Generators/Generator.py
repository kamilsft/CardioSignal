import torch
import torch.nn as nn
from Models.cycle_gan.Generators.Transformer import Transformer

class Generator(nn.Module):
    def __init__(self, input_channels=1, conv_dim=128, sequence_length=48000):
        super(Generator, self).__init__()

        # # 1. Encoder (Conv1D)
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(input_channels, conv_dim, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
        #     nn.ReLU()
        # )

        # 1. Downsampler to reduce sequence length before we pass it to the transformer as it is very expensive to process long sequences and get Run time errors
        self.downsampler = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=9, stride=4, padding=4), # 48000 → 12000
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2), # 12000 → 3000
            nn.ReLU(),
            nn.Conv1d(64, conv_dim, kernel_size=5, stride=3, padding=2), # 3000 → 1000
            nn.ReLU()
        )

        # 2. Transformer
        self.transformer = Transformer(input_dim=conv_dim)

        # 3. Decoder (Conv1D)
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(conv_dim, input_channels, kernel_size=7, padding=3),
        #     nn.Tanh()  # Output between -1 and 1
        # )

        # 3. Upsampler to restore original resolution
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='linear', align_corners=True), # 1000 → 3000
            nn.Conv1d(conv_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='linear', align_corners=True), # 3000 → 12000
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='linear', align_corners=True), # 12000 → 48000
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
            nn.Tanh()
)

    def forward(self, x):
        # x = self.encoder(x)
        x = self.downsampler(x) # we have[B, 1, 48000] --> [B, 128, 1000] after downsampling
        x = x.permute(0, 2, 1) # changing the shape for the transformer and also reordering the dimension [batch_size, channels, sequence_length] → [batch_size, sequence_length, channels]
        x = self.transformer(x)
        x = x.permute(0, 2, 1) # changing the shape back for the decoder and also reordering the dimension
        # x = self.decoder(x)
        x = self.upsampler(x) # we have[B, 128, 1000] --> [B, 1, 48000] after upsampling
        return x
        
if __name__ == "__main__":
    model = Generator()
    test_input = torch.randn(4,1,48000) # batch_size, channels, sequence_length just for testing 
    test_output = model(test_input)
    print("Generator output shape:", test_output.shape)  