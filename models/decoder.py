import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super().__init__()
        self.deconv1 = self.make_layer(latent_dim // 4, 512)  # 4
        self.deconv2 = self.make_layer(512, 512)  # 8
        self.deconv3 = self.make_layer(512, 256)  # 16
        self.deconv4 = self.make_layer(256, 256)  # 32
        self.deconv5 = self.make_layer(256, 128)  # 64
        self.deconv6 = self.make_layer(128, 128)  # 128
        self.deconv7 = self.make_layer(128, 64)  # 256
        self.deconv8 = self.make_layer(64, 64)  # 512
        self.deconv9 = self.make_layer(64, 3, is_last=True)  # 512

    def forward(self, x):
        x = x.view(x.size(0), -1, 2, 2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        # x = self.deconv8(x)
        x = self.deconv9(x)
        return x

    def make_layer(
        self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, is_last=False
    ):
        layer = []
        layer.append(nn.ReflectionPad2d(padding))
        layer.append(nn.Conv2d(in_dim, out_dim, kernel_size, stride, bias=False))
        if not is_last:
            layer.append(nn.BatchNorm2d(out_dim))
            layer.append(nn.LeakyReLU(0.2, True))
            layer.append(nn.Upsample(scale_factor=2))
        else:
            layer.append(nn.Tanh())
        layer = nn.Sequential(*layer)
        return layer


def test():
    import torch

    model = Decoder()
    x = torch.rand([1, 2048])
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
