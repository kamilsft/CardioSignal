import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, num_layers=3, dropout=0.1, ff_dim=256):
        super(Transformer, self).__init__()

 # we will use 3 encoder layers as it is a common choice for many tasks
        # define one encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim,
            dropout=dropout, 
            batch_first=True
        )

        # now we are stacking each encoder layer to create a full transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
    
    def forward(self, x):
        return self.transformer_encoder(x)

if __name__ == "__main__":
    # Example usage
    model = Transformer()
    sample_input = torch.randn(4, 48000, 128)  # (batch_size, sequence_length, input_dim)
    output = model(sample_input)
    print(output.shape)  # Should be (4,500,128))