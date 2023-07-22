import torch
import torch.nn as nn
import torch.nn.functional as F


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


def test():
    model = StyleGenerator()
    mapper = MappingNet()
    content = torch.rand(1, 512, 4, 4)
    w = torch.rand(1, 512)
    noise = torch.rand(1, 512, 1, 1)

    y = model(content, w, noise)
    pass


if __name__ == "__main__":
    test()
