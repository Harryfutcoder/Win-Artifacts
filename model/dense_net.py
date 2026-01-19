import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(self, num_blocks=3, num_layers_per_block=4, growth_rate=12, input_channels=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv1d(input_channels, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False)
        self.blocks = nn.ModuleList()
        num_channels = growth_rate * 2

        for i in range(num_blocks):
            self.blocks.append(DenseBlock(num_layers_per_block, num_channels, growth_rate))
            num_channels += num_layers_per_block * growth_rate
            if i != num_blocks - 1:
                self.blocks.append(TransitionLayer(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.bn = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.hidden1 = nn.Linear(num_channels, 64)
        self.hidden2 = nn.Linear(64, 32)

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.fc(x)
        return x
