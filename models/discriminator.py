import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = self._conv(3, 64, first=True)  # 1/2
        self.conv2 = self._conv(64, 64)  # 1/4
        self.conv3 = self._conv(64, 128)  # 1/8
        self.conv4 = self._conv(128, 256)  # 1/16
        self.conv5 = self._conv(256, 512)  # 1/32
        self.conv6 = self._conv(512, 512)  # 1/64
        self.conv7 = self._conv(512, 512)  # 1/128
        self.conv8 = self._conv(512, 512)  # 1 / 256
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 1, 2, 1, 0, bias=False), nn.Sigmoid()
        )  # 1/512

    def _conv(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, first=False):
        if not first:
            layer = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
                nn.Mish(inplace=True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.Mish(inplace=True),
            )
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.shape[0], -1)
        return x

