import torch
import torch.nn as nn

from models.resnet_block import UpsampleResNetBlock


class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.conv(x)
        weights = weights.view(weights.shape[0], 1, -1)
        weights = self.softmax(weights)
        weights = weights.view_as(x)
        x = x * weights
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 1024))
        # self.attention = AttentionLayer(3)  # Attention 레이어 추가
        self.conv1 = UpsampleResNetBlock(1024, 512)  # 1/2
        self.conv2 = UpsampleResNetBlock(512, 512)  # 1/4
        self.conv3 = UpsampleResNetBlock(512, 256)  # 1/8
        self.conv4 = UpsampleResNetBlock(256, 256)  # 1/16
        self.conv5 = UpsampleResNetBlock(256, 3)  # 1/32
        # self.conv6 = UpsampleResNetBlock(128, 3)  # 1/64
        # self.conv7 = UpsampleResNetBlock(128, 64)  # 1/128
        # self.conv8 = UpsampleResNetBlock(64, 3)  # 1 / 256
        # self.conv9 = UpsampleResNetBlock(64, 3)
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(3, 3, 3, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.attention(x)
        x = self.last_layer(x)
        return x


def test():
    x = torch.rand(1, 3, 512, 512)
    model = Discriminator()
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
