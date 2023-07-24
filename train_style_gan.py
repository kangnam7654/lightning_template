from pipelines.style_gan import StyleGanPipeline
from models.style_gan import StyleGenerator, MappingNet
from models.discriminator import StyleDiscriminator



import torch.nn as nn
from pipelines.gan import GANPipeline
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodules.image_module import CustomImageDataset
from models.discriminator import Discriminator
from models.generator import Generator
import torchvision
import torchvision.transforms as transforms
from models.feature_extractors import ResNet50_
import torch 

torch.set_float32_matmul_precision('high')

nc = 3
nz = 100
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    mapper = MappingNet(512)
    discriminator = StyleDiscriminator()
    generator = StyleGenerator()
    
    # weights_init(discriminator)
    # weights_init(generator)

    pipeline = StyleGanPipeline(
        discriminator=discriminator, generator=generator, mapper=mapper, lr_d=2e-4, lr_g=2e-4, n_critic=1
    )
    
    dataset = CustomImageDataset(
        # data_dir="/home/kangnam/datasets/images/cocosets/train2017/",
        # data_dir="/home/kangnam/datasets/images/kangnam_resize/",
        data_dir="/home/kangnam/datasets/images/sakimichan/",
        return_label=False,
        resolution=(256, 256),
        data_length=None,
        data_repeat=100
    )

    # print(dataset.__len__())
    dataloder = DataLoader(dataset, batch_size=2)
    trainer = pl.Trainer(max_epochs=100000000)
    trainer.fit(model=pipeline, train_dataloaders=dataloder)


if __name__ == "__main__":
    main()
