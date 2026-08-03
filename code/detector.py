import torch
import torch.nn as nn
from config import Config


class Configurable1DCNN(nn.Module):

    def __init__(self, config=None):
        super(Configurable1DCNN, self).__init__()

        if config is None:
            config = Config.DETECTOR_CONFIG

        in_channels = config.get("input_channels", 1)
        conv_filters = config.get("conv_filters", [16, 32, 64, 128])
        kernel_sizes = config.get("kernel_sizes", [7, 5, 5, 3])
        fc_units = config.get("fc_units", 64)
        dropout_rate = config.get("dropout", 0.3)

        layers = []
        current_channels = in_channels

        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Intermediate adaptive pooling preserves temporal pulse structure without exploding linear weights
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        flattened_dim = conv_filters[-1] * 64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
