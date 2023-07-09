import torch
import torch.nn as nn
from pipelines.gan import GANPipeline
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodules.image_module import CustomImageDataset
from models.discriminator import Discriminator
from models.generator import Generator
import torchvision
import torchvision.transforms as transforms


def main():
    discriminator = Discriminator()
    generator = Generator(1)
    # dataset = CustomImageDataset(
    #     data_dir="/home/kangnam/datasets/images/1_kangnam_diffusion/",
    #     return_label=False,
    #     resolution=(64,64)
    # )
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageNet(root='./data/imagenet/train', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageNet(root='./data/imagenet/valid', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)
    pipeline = GANPipeline(
        discriminator=discriminator, generator=generator, lr_d=1e-5, lr_g=1e-5, n_critic=2
    )

    # print(dataset.__len__())
    # dataloder = DataLoader(dataset, batch_size=2)
    trainer = pl.Trainer()
    trainer.fit(model=pipeline, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
