import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim=39):
        super().__init__()
        self.deconv1 = self._make_layer(latent_dim, 512, 4, 1, 0)  # 2
        # self.deconv1_2 = self._make_layer(512, 512)
        self.deconv2 = self._make_layer(512, 512)  # x 4
        self.deconv3 = self._make_layer(512, 512)  # x 8
        self.deconv4 = self._make_layer(512, 256)  # x 16
        self.deconv5 = self._make_layer(256, 128)  # x 32
        self.deconv6 = self._make_layer(128, 64)  # x 64
        self.deconv7 = self._make_layer(64, 64)  # x 128
        self.deconv8 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())


    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        return x

    def _make_layer(
        self,
        in_dim,
        out_dim,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, out_dim, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_dim),
            nn.Mish(inplace=True),
            # nn.Dropout(p=0.2),
        )
        return conv




class Imitator_(nn.Module):
    def __init__(self, latent_dim=39):
        super().__init__()
        self.deconv1 = self._make_layer(latent_dim, 512, 4, 1, 0)  # 2
        self.deconv2 = self._make_layer(512, 512)  # x 4
        self.deconv3 = self._make_layer(512, 512)  # x 8
        self.deconv4 = self._make_layer(512, 256)  # x 16
        self.deconv5 = self._make_layer(256, 128)  # x 32
        self.deconv6 = self._make_layer(128, 64)  # x 64
        self.deconv7 = self._make_layer(64, 64)  # x 128
        self.deconv8 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        return x

    def _make_layer(
        self,
        in_dim,
        out_dim,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, out_dim, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_dim),
            nn.Mish(inplace=True),
            # nn.Dropout(p=0.2),
        )
        return conv


def test():
    import torch

    model = Decoder()
    x = torch.rand([1, 39, 1, 1])
    y = model(x)
    # print(y.shape)


if __name__ == "__main__":
    test()
