import torch.nn as nn
import torchvision.models as models

class ShapeColorClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)