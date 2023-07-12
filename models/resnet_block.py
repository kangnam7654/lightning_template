import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.unlinearity = nn.LeakyReLU(True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x.clone()
        x = self.conv1(x)
        x = self.unlinearity(x)
        x = self.conv2(x)
        x += self.shortcut(residual)
        x = self.unlinearity(x)
        return x


class UpsampleResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, is_last=False):
        super().__init__()
        self.unlinearity = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, out_dim, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.conv2 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()
        
        if not is_last:
            self.upsample_layer = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_dim, out_dim, 3, 1),
            )
        else:
            self.upsample_layer = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_dim, out_dim, 3, 1),
                nn.Tanh()
            )


    def forward(self, x):
        residual = x.clone()
        x = self.conv1(x)
        x = self.unlinearity(x)
        x = self.conv2(x)
        x += self.shortcut(residual)
        x = self.unlinearity(x)
        x = self.upsample_layer(x)

        return x
