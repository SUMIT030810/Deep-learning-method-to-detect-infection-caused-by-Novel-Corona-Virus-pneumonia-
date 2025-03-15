import torch
import torch.nn as nn


# Depthwise Separable Convolution block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# MobileNet Model
class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),  # 112x112x64
            DepthwiseSeparableConv(64, 128, stride=2),  # 56x56x128
            DepthwiseSeparableConv(128, 128, stride=1),  # 56x56x128
            DepthwiseSeparableConv(128, 256, stride=2),  # 28x28x256
            DepthwiseSeparableConv(256, 256, stride=1),  # 28x28x256
            DepthwiseSeparableConv(256, 512, stride=2),  # 14x14x512
            *[DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5)],  # 5 blocks 14x14x512
            DepthwiseSeparableConv(512, 1024, stride=2),  # 7x7x1024
            DepthwiseSeparableConv(1024, 1024, stride=1)  # 7x7x1024
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, 1024]
        x = self.fc(x)
        return x


def MobileNet_Model(num_classes):
    model = MobileNet(num_classes=num_classes)
    name = "MobileNet"
    return model, name
