import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    """Dual Convolution Block for 1D U-Net."""
    def __init__(self, in_ch, out_ch, kernel_size=3, use_batch_norm=True, dropout=0.1):
        super(ConvBlock1D, self).__init__()
        padding = kernel_size // 2
        
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Configurable1DUNet(nn.Module):
    """Dynamic 1D U-Net for Signal Denoising."""
    def __init__(self, config):
        super(Configurable1DUNet, self).__init__()
        
        in_channels = config.get("input_channels", 1)
        features = config.get("features", [32, 64, 128, 256])
        kernel_size = config.get("kernel_size", 3)
        use_bn = config.get("batch_norm", True)
        dropout = config.get("dropout", 0.1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.ModuleList()

        # Encoder Path
        curr_in = in_channels
        for feat in features:
            self.encoders.append(ConvBlock1D(curr_in, feat, kernel_size, use_bn, dropout))
            curr_in = feat

        # Bottleneck Path
        bottleneck_dim = features[-1] * 2
        self.bottleneck = ConvBlock1D(features[-1], bottleneck_dim, kernel_size, use_bn, dropout)

        # Decoder Path
        reversed_features = list(reversed(features))
        curr_in = bottleneck_dim
        for feat in reversed_features:
            self.upsample.append(nn.ConvTranspose1d(curr_in, feat, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock1D(feat * 2, feat, kernel_size, use_bn, dropout))
            curr_in = feat

        # Final Reconstruction Layer
        self.final_conv = nn.Conv1d(features[0], in_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(len(self.decoders)):
            x = self.upsample[idx](x)
            skip = skip_connections[idx]

            # Adjust dimension alignment if length mismatch occurs due to odd dimensions
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)

        return self.final_conv(x)

