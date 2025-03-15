import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        self.stride = stride

        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)

        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.se = SqueezeExcitation(hidden_dim, reduction=int(1 / se_ratio))

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.swish(self.bn0(self.expand_conv(x)))
        x = self.swish(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + identity
        return x
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),
            MBConvBlock(16, 24, expand_ratio=6, stride=2),
            MBConvBlock(24, 40, expand_ratio=6, stride=2),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2),
            MBConvBlock(192, 320, expand_ratio=6, stride=1)
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# class EfficientNetModel():
#     def __init__(self,num_classes):
#         self.num_classes=num_classes
#     def build_model(self):
#         model = EfficientNet(self.num_classes)
#         name='EfficientNet'
#         return model,name

def EfficientNetModel(num_classes):
    model=EfficientNet(num_classes=num_classes)
    name="EfficientNet"
    return model,name
