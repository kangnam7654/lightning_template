import torch
import torch.nn as nn

from models.resnet_block import ResNetBlock

class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        weights = self.conv(x)
        weights = weights.view(weights.shape[0], 1, -1)
        weights = self.softmax(weights)
        weights = weights.view_as(x)
        x = x * weights
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = ResNetBlock(3, 64, 2)  # 1/2
        self.conv2 = ResNetBlock(64, 64, 2)  # 1/4
        self.conv3 = ResNetBlock(64, 128, 2)  # 1/8
        self.conv4 = ResNetBlock(128, 128, 2)  # 1/16
        self.conv5 = ResNetBlock(128, 256, 2)  # 1/32
        # self.conv6 = ResNetBlock(256, 256, 2)  # 1/64
        # self.conv7 = ResNetBlock(256, 512, 2)  # 1/128
        # self.conv8 = ResNetBlock(512, 1024, 2)  # 1 / 256
        # self.conv9 = ResNetBlock(512, 1024, 2)
        self.attention = AttentionLayer(256)  # Attention 레이어 추가
        self.fc = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        
        x = self.attention(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test():
    x = torch.rand(1, 3, 512, 512)
    model = Discriminator()
    y = model(x)
    print(y.shape)
if __name__ == "__main__":
    test()