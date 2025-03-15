import torch
import torch.nn as nn

# Define the AlexNet architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]
        x = self.classifier(x)
        return x

# Define a function to return the model and its name
def AlexNet_Model(num_classes):
    model = AlexNet(num_classes=num_classes)
    name = "AlexNet"
    return model, name
"""
# Example Usage
if __name__ == "__main__":
    num_classes = 3
    model, name = AlexNet_Model(num_classes)
    print(f"Model name: {name}")

    # Test with random input
    x = torch.randn(1, 3, 227, 227)  # AlexNet expects 227x227 input
    output = model(x)
    print(f"Output shape: {output.shape}")
"""