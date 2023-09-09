import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_block import ResNetBlock


class AdaIN(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.style_affine = nn.Linear(512, 2 * n_ch)
        self.norm = nn.InstanceNorm2d(n_ch)

    def forward(self, w, content):  # content (bs, ch, h, w)
        style = self.style_affine(w)  # (bs, 2 * ch)
        style = style.unsqueeze(2).unsqueeze(3)
        sigma, mu = style.chunk(2, 1)

        normalized = self.norm(content)
        out = sigma * normalized + mu
        return out


class InitialSynthesisLayer(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.noise_affine = nn.Conv2d(512, out_dim, 1, 1, 0)
        self.adain = AdaIN(out_dim)

    def forward(self, constant, mapped, noise):
        noise = self.noise_affine(noise)
        content = constant + noise
        content = self.adain(mapped, constant)
        return content


class SynthesisLayer(nn.Module):
    def __init__(self, in_dim, out_dim, upsample=False):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            )
        else:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.noise_affine = nn.Conv2d(512, out_dim, 1, 1, 0)
        self.adain = AdaIN(out_dim)

    def forward(self, content, mapped, noise):
        content = self.conv1(content)

        noise = self.noise_affine(noise)
        content = content + noise
        content = self.adain(mapped, content)
        return content


class SynthesisBlock(nn.Module):
    def __init__(self, in_dim, out_dim, initial=False):
        super().__init__()
        if initial:
            self.layer1 = InitialSynthesisLayer(out_dim)
        else:
            self.layer1 = SynthesisLayer(in_dim, out_dim, upsample=True)
        self.layer2 = SynthesisLayer(out_dim, out_dim, upsample=False)

    def forward(self, content, mapped, noise):
        content = self.layer1(content, mapped, noise)
        content = self.layer2(content, mapped, noise)
        return content


class StyleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SynthesisBlock(512, 512, initial=True)  # 4
        self.layer2 = SynthesisBlock(512, 256)  # 8
        self.layer3 = SynthesisBlock(256, 256)  # 16
        self.layer4 = SynthesisBlock(256, 128)  # 32
        self.layer5 = SynthesisBlock(128, 64)  # 64
        self.layer6 = SynthesisBlock(64, 64)  # 128
        self.layer7 = SynthesisBlock(64, 3)  # 256

    def forward(self, content, mapped, noise):
        content = self.layer1(content, mapped, noise)
        content = self.layer2(content, mapped, noise)
        content = self.layer3(content, mapped, noise)
        content = self.layer4(content, mapped, noise)
        content = self.layer5(content, mapped, noise)
        content = self.layer6(content, mapped, noise)
        content = self.layer7(content, mapped, noise)
        return content


class MappingNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(in_dim, 512)
        self.fc3 = nn.Linear(in_dim, 512)
        self.fc4 = nn.Linear(in_dim, 512)
        self.fc5 = nn.Linear(in_dim, 512)
        self.fc6 = nn.Linear(in_dim, 512)
        self.fc7 = nn.Linear(in_dim, 512)
        self.fc8 = nn.Linear(in_dim, 512)

    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.fc5(z)
        z = self.fc6(z)
        z = self.fc7(z)
        z = self.fc8(z)
        return z


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
        self.conv6 = ResNetBlock(256, 256, 2)  # 1/64
        self.conv7 = ResNetBlock(256, 512, 2)  # 1/128
        self.conv8 = ResNetBlock(512, 512, 2)  # 1 / 256
        # self.conv9 = ResNetBlock(512, 1024, 2)
        # self.attention = AttentionLayer(1024)  # Attention 레이어 추가
        self.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        # x = self.conv9(x)

        # x = self.attention(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        size = x.size()
        sub_group_size = min(size[0], self.group_size)
        if size[0] % sub_group_size != 0:
            sub_group_size = size[0]
        G = size[0] // sub_group_size
        if sub_group_size > 1:
            y = x.view(-1, sub_group_size, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, sub_group_size, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return torch.cat([x, y], dim=1)


class LastStyleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.minibatch_std = MinibatchStdDev()
        self.conv1 = nn.Conv2d(in_dim + 1, out_dim, 3, 1, 1)
        self.l_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 4, 1, 0)
        self.l_relu2 = nn.LeakyReLU(inplace=True)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.minibatch_std(x)
        x = self.conv1(x)
        x = self.l_relu1(x)
        x = self.conv2(x)
        x = self.l_relu2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class StyleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, initial=False):
        super(StyleBlock, self).__init__()
        if initial:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_dim, 16, 1, 1, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.LeakyReLU(inplace=False),
            )
            self.l_relu1 = nn.LeakyReLU(inplace=True)
            self.conv2 = nn.Conv2d(16, out_dim, 3, 1, 1)

        else:
            self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
            self.l_relu1 = nn.LeakyReLU(inplace=True)
            self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.l_relu2 = nn.LeakyReLU(inplace=True)
        self.downsample = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.l_relu1(x)
        x = self.conv2(x)
        x = self.l_relu2(x)
        x = self.downsample(x)
        return x


class StyleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = StyleBlock(3, 32, initial=True)  # 1/2
        self.layer2 = StyleBlock(32, 64)  # 1/4
        self.layer3 = StyleBlock(64, 128)  # 8
        self.layer4 = StyleBlock(128, 256)  # 16
        self.layer5 = StyleBlock(256, 512)  # 1/32
        self.layer6 = StyleBlock(512, 512)  # 1/64
        self.layer7 = LastStyleBlock(512, 512)  # 1/256

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


def test():
    x = torch.rand(1, 3, 256, 256)
    model = StyleDiscriminator()
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
