from pipelines.gan import GANPipeline
from pipelines.auto_encoder import ImitatorPipeline
from models.discriminator import Discriminator
from models.feature_extractors import (
    ResNet50FeatureExtractor,
    ResNet152FeatureExtractor,
)
from models.decoder import Decoder
from datamodules.image_module import CustomImageDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from utils.load_mnist import read_mnist
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class TempGenerator(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer3 = nn.Sequential(nn.Linear(512, 784, bias=False))

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), 28, 28)
        return x


class TempDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 512, bias=False),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, affine=False),
            nn.LeakyReLU(0.2, True),
        )
        self.layer3 = nn.Sequential(nn.Linear(256, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.images, self.labels = read_mnist("data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.preprocess(image)
        return image, label

    def preprocess(self, image):
        image = torch.from_numpy(image)  # shape == (28, 28)
        image = (image / 127.5) - 1
        return image


def to_image(image):
    image = image[0].clone().detach().cpu()
    # image = image.numpy()
    # image = np.array(image, dtype=np.uint8)
    return image


def main():
    dataset = CustomImageDataset(
        data_dir="/home/kangnam/datasets/image/1_kangnam_diffusion",
        data_length=None,
        data_repeat=100,
        return_label=False,
    )
    dataloder = DataLoader(dataset, batch_size=2, num_workers=8)
    discriminator = torch.compile(Discriminator())
    generator = torch.compile(Decoder(4))
    gan = GANPipeline(
        generator=generator,
        discriminator=discriminator,
        manual_ckpt_save_path="./gan.ckpt",
        lr_d=1e-8,
        lr_g=1,
        n_critic=1,
        image_logging_interval=100,
    )
    ae = ImitatorPipeline(discriminator, generator, lr=1e-3, show=True)
    trainer = pl.Trainer(min_epochs=100, devices=1, accelerator="auto")
    trainer.fit(model=gan, train_dataloaders=dataloder)

class Rectified(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        out = torch.where(x >= 0, torch.tensor(1.), torch.tensor(-1.))
        return out
    
def main2():
    discriminator = torch.compile(TempDiscriminator())
    generator = torch.compile(TempGenerator(10))

    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)

    dataset = MNIST()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    epochs = 100
    for epoch in range(epochs):
        for idx, batch in enumerate(loader):
            real_images, number = batch  # number: torch.Tensor

            number = number.unsqueeze(1)
            bs = real_images.size(0)
            z = torch.concat([number, torch.randn(bs, 9)], dim=-1)
            fake_images = generator(z)

            real_labels = torch.ones((bs, 1))
            real_labels = real_labels.type_as(real_images)
            fake_labels = torch.zeros((bs, 1))
            fake_labels = fake_labels.type_as(real_images)

            real_preds = discriminator(real_images)
            fake_preds = discriminator(fake_images)

            real_loss = criterion(real_preds, real_labels)
            fake_loss = criterion(fake_preds, fake_labels)
            d_loss = (real_loss + fake_loss) * 0.5

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ======= GENERATOR ====
            z = torch.concat([number, torch.randn(bs, 9)], dim=-1)
            fake_images = generator(z)

            real_labels = torch.ones((bs, 1))
            real_labels = real_labels.type_as(real_images)

            fake_preds = discriminator(fake_images)
            g_loss = criterion(fake_preds, real_labels)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            image = to_image(fake_images)
            if idx % 50 == 0:
                image = np.hstack([real_images[0], image])
                image = cv2.resize(image, (448, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("", image)
                cv2.waitKey(1)
                print(
                    f"epoch: {epoch}, step: {idx}, g_loss: {round(g_loss.item(), 4)}, d_loss: {round(d_loss.item(), 4)}, label: {int(number[0])}"
                )


def test():
    images, labels = read_mnist("data")
    sample = images[0]

    cv2.imshow("", sample)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == "__main__":
    main2()
    # test()
