import torch
import torch.nn as nn


class DoubleConv1D(nn.Module):
    """Dual Convolution Block with Batch Normalization and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True, dropout=0.1):
        super(DoubleConv1D, self).__init__()
        padding = kernel_size // 2
        
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Configurable1DUNet(nn.Module):
    """1D U-Net Architecture for Signal Denoising and Reconstruction."""
    def __init__(self, config):
        super(Configurable1DUNet, self).__init__()
        in_channels = config.get("input_channels", 1)
        out_channels = config.get("out_channels", 1)
        features = config.get("features", [32, 64, 128, 256])
        kernel_size = config.get("kernel_size", 3)
        batch_norm = config.get("batch_norm", True)
        dropout = config.get("dropout", 0.1)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        curr_in = in_channels
        for feature in features:
            self.encoders.append(
                DoubleConv1D(curr_in, feature, kernel_size, batch_norm, dropout)
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            curr_in = feature

        self.bottleneck = DoubleConv1D(
            features[-1], features[-1] * 2, kernel_size, batch_norm, dropout
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        curr_in = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose1d(curr_in, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                DoubleConv1D(feature * 2, feature, kernel_size, batch_norm, dropout)
            )
            curr_in = feature

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder pass with skip concatenation
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]

            # Padding adjustment if shape mismatch due to odd dimensions
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i](x)

        return self.final_conv(x)
