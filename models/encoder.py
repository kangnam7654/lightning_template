import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, out_dim=39):
        super().__init__()
        self.conv1 = self._conv(3, 64, 7, 4, 1)  # 1/4
        self.conv2 = self._conv(64, 64)  # 1/8
        self.conv3 = self._conv(64, 128)  # 1/16
        self.conv4 = self._conv(128, 256)  # 1/32
        self.conv5 = self._conv(256, 512)  # 1/64
        self.conv6 = self._conv(512, 512)  # 1/128
        self.conv7 = self._conv(512, 512)  # 1/256
        self.conv8 = self._conv(512, out_dim, last=True)  # 1/512

    def _conv(self, in_dim, out_dim, kernel_size=3, stride=2, padding=1, last=False):
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        if last:
            layer = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
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
        return x
