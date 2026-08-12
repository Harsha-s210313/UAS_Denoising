import torch
import torch.nn as nn


class SignalDetectorCNN(nn.Module):
    """
    Stage 1 Binary Classification Model.
    Detects whether an acoustic pulse/echo signature is present in the signal window.
    """
    def __init__(self, config):
        super(SignalDetectorCNN, self).__init__()
        in_channels = config.get("input_channels", 1)
        filters = config.get("conv_filters", [16, 32, 64, 128])
        kernel_sizes = config.get("kernel_sizes", [7, 5, 5, 3])
        fc_units = config.get("fc_units", 64)
        dropout = config.get("dropout", 0.1)

        layers = []
        curr_channels = in_channels

        for num_filters, k_size in zip(filters, kernel_sizes):
            layers.append(
                nn.Conv1d(
                    curr_channels,
                    num_filters,
                    kernel_size=k_size,
                    padding=k_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_channels = num_filters

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(filters[-1], fc_units),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        out = self.classifier(x)
        return out.squeeze(-1)
