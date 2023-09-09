import torch.nn as nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval()
        self.model = nn.Sequential(*list(vgg.features.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(7)

        self.block_idx = 1
        self.conv_idx = 1
        self.relu_idx = 1

    def get_features(self, x):
        features = {}
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features[f"conv{self.block_idx}_{self.conv_idx}"] = x
                self.conv_idx += 1
            if isinstance(layer, nn.ReLU):
                features[f"relu{self.block_idx}_{self.relu_idx}"] = x
                self.relu_idx += 1
            if isinstance(layer, nn.MaxPool2d) or isinstance(
                layer, nn.AdaptiveAvgPool2d
            ):
                features[f"pool{self.block_idx}"] = x
                self.block_idx += 1
                if self.block_idx > 5:
                    self.block_idx = 1
                self.conv_idx = 1
                self.relu_idx = 1
        return features

    def forward(self, x):
        x = self.model(x)
        return x


def test():
    import torch

    model = VGG16FeatureExtractor()
    x = torch.rand(1, 3, 256, 256)
    result = model.get_features(x)
    pass


if __name__ == "__main__":
    test()
