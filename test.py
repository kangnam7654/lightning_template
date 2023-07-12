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
from models.discriminator import Discriminator
from models.generator import Generator


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
    image = (image + 1) * 127.5
    image = torch.permute(image, (1, 2, 0))
    image = image.numpy()
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = image.numpy()
    # image = np.array(image, dtype=np.uint8)
    return image


def main():
    dataset = CustomImageDataset(
        data_dir="/home/kangnam/datasets/images/cocosets/train2017/",
        data_length=None,
        data_repeat=1,
        return_label=False,
    )
    dataloder = DataLoader(dataset, batch_size=16, num_workers=8)
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
        out = torch.where(x >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        return out

def main2():

    generator = Generator(50)

    generator = generator.cuda()
    
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)

    dataset = CustomImageDataset(
        data_dir=("/home/kangnam/datasets/images/cocosets/train2017/"),
        resolution=(256, 256),
        return_label=False,
        data_length=1,
        data_repeat=1000000
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.L1Loss()

    epochs = 100
    for epoch in range(epochs):
        for idx, batch in enumerate(loader):
            real_images = batch[0]  # number: torch.Tensor
            real_images = real_images.cuda()
            
            bs = real_images.size(0)
            # ======= GENERATOR ====
            z = torch.ones(bs, 50)
            z = z.cuda()
            fake_images = generator(z)

            g_loss = criterion(fake_images, real_images)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            fake_ = to_image(fake_images)
            real_ = to_image(real_images)
            
            image = np.hstack([real_, fake_])
            cv2.imshow("", image)
            cv2.waitKey(1)
            if idx % 10 == 0:
                # image = np.transpose(image, (1, 2, 0))
                # image = cv2.resize(image, (256, 128), interpolation=cv2.INTER_NEAREST)

                print(
                    f"epoch: {epoch}, step: {idx}, g_loss: {round(g_loss.item(), 4)}"
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
