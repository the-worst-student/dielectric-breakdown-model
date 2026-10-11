import torch.nn as nn
import torch as torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + identity)


class DBMResidualCNN(nn.Module):
    def __init__(self, number_of_targets=4):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2),

            ResidualBlock(16, 32),
            nn.MaxPool2d(2),

            ResidualBlock(32, 64),
            nn.MaxPool2d(2),

            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
        )

        self.average_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.maximum_pool = nn.AdaptiveMaxPool2d((4, 4))

        self.shared_layers = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10),
                    nn.Linear(32, 1),
                )
                for _ in range(number_of_targets)
            ]
        )

    def encode(self, x):
        features = self.encoder(x)

        pooled = torch.cat(
            [
                self.average_pool(features).flatten(1),
                self.maximum_pool(features).flatten(1),
            ],
            dim=1,
        )

        return self.shared_layers(pooled)

    def predict_embedding(self, embedding):
        return torch.cat(
            [head(embedding) for head in self.heads],
            dim=1,
        )

    def forward(self, x):
        return self.predict_embedding(self.encode(x))