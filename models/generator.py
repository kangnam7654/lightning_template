import torch
import torch.nn as nn

from models.resnet_block import UpsampleResNetBlock


class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.q = nn.Linear(in_channels, in_channels * 3, bias=False)
        self.k = nn.Linear(in_channels, in_channels * 3, bias=False)
        self.v = nn.Linear(in_channels, in_channels * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1) # 
        q = self.q(x) # (bs, 3c, 1)
        k = self.k(x)# (bs, 3c, 1)
        v = self.v(x)# (bs, 3c, 1)
        qk = torch.einsum("bij, bkl -> bik", qk) # (bs, 3c, 3c)
        qk = self.softmax(qk)
        qkv = qk
        weights = weights.view(weights.shape[0], 1, -1)
        weights = self.softmax(weights)
        weights = weights.view_as(x)
        x = x * weights
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0)) # 4
        self.conv2 = UpsampleResNetBlock(512, 512)  # 8
        self.conv3 = UpsampleResNetBlock(512, 256)  # 16
        self.conv4 = UpsampleResNetBlock(256, 128)  # 32
        self.conv5 = UpsampleResNetBlock(128, 128)  # 64
        self.conv6 = UpsampleResNetBlock(128, 64)  # 128
        self.conv7 = UpsampleResNetBlock(64, 3, is_last=True)  # 256
        # self.conv8 = UpsampleResNetBlock(128, 3, is_last=True)  # 512
        # self.conv9 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3, 1, 0, bias=False), nn.Tanh())
        # self.conv8 = UpsampleBlock(64, 3)  # 1 / 256
        # self.conv9 = UpsampleResNetBlock(64, 3)
        # self.last_layer = nn.Sequential(nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.last_layer(x)
        return x


def test():
    x = torch.rand(1, 3, 512, 512)
    model = Discriminator()
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
