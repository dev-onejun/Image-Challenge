from torchvision.models import (
    resnet152,
    ResNet152_Weights,
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)
from torch import nn


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.model(x)


class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()

        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(1280, 10)

    def forward(self, x):
        return self.model(x)


class MixNet(nn.Module):
    def __init__(self):
        super(MixNet, self).__init__()

        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048, 1280)

        self.efficientnet = efficientnet_v2_l(
            weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1
        )
        self.classifier = self.efficientnet.classifier
        self.classifier[1] = nn.Linear(1280, 10)
        self.efficientnet.classifier = nn.Sequential(nn.Identity())

    def forward(self, x):
        resnet_output = self.resnet(x)
        efficientnet_output = self.efficientnet(x)

        out = resnet_output + efficientnet_output
        self.classifier(out)

        return out


if __name__ == "__main__":
    model = ResNet152()
    print(model)
