import cv2
import numpy as np
import torch
import torch.nn as nn


class TestDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self.first_layer(2048 // 4, 512)
        self.layer2 = self._layer(512, 512)  # 4
        self.layer3 = self._layer(512, 256)  # 8
        self.layer4 = self._layer(256, 256)  # 16
        self.layer5 = self._layer(256, 128)  # 32
        self.layer6 = self._layer(128, 128)  # 64
        self.layer7 = self._layer(128, 64)  # 128
        self.layer8 = self._layer(64, 64)
        self.layer9 = self.last_layer(64, 3)

    # 512

    def forward(self, x):
        x = x.view(x.size(0), -1, 2, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x

    def first_layer(self, in_dim, out_dim):
        layer = [
            # nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_dim, out_dim, 3, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, True),
        ]
        layer = nn.Sequential(*layer)
        return layer

    def last_layer(self, in_dim, out_dim):
        layer = [
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_dim, out_dim, 3, 1, bias=False),
            # nn.BatchNorm2d(out_dim),
            nn.Tanh(),
        ]
        layer = nn.Sequential(*layer)
        return layer

    def _layer(self, in_dim, out_dim):
        layer = []
        layer.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        layer.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, bias=False))
        layer.append(nn.BatchNorm2d(out_dim))
        layer.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layer.append(nn.Upsample(scale_factor=2))
        layer = nn.Sequential(*layer)
        return layer


def preprocess(image_path):
    image_raw = cv2.imread(image_path)
    image_raw = cv2.resize(image_raw, (512, 512))
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, [2, 0, 1])
    image = (image / 127.5) - 1
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.float()
    return image, image_raw


def tensor_to_image(t: torch.Tensor):
    tt = t.clone().detach().cpu().squeeze(0).numpy()
    tt = np.transpose(tt, (1, 2, 0))
    tt = tt * 0.5 + 0.5
    tt = np.clip(tt, 0, 1)
    tt = tt * 255
    tt = tt.astype(np.uint8)
    tt = cv2.cvtColor(tt, cv2.COLOR_RGB2BGR)
    return tt


def main():
    model = TestDecoder()
    image_path = "data/sample/image/cat.png"
    image, raw = preprocess(image_path)
    epochs = 10000

    model = model.cuda()
    image = image.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.randn((1, 2048, 1, 1)).cuda()

    for epoch in range(epochs):
        for i in range(10000):
            reconstructed = model(x)
            loss = criterion(reconstructed, image)

            image_ = tensor_to_image(reconstructed)
            to_show = np.concatenate([raw, image_], axis=1)
            cv2.imshow("", to_show)
            cv2.waitKey(1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f"epoch: {epoch}, step: {i}, loss: {loss}")


if __name__ == "__main__":
    main()
