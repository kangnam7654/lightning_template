
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


class Generator_(nn.Module):
    def __init__(self):
        super(Generator_, self).__init__()
        self.layer1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        self.layer2 = self.make_layer(512, 512) # 8
        self.layer3 = self.make_layer(512, 256) # 16
        self.layer4 = self.make_layer(256, 256) # 32
        self.layer5 = self.make_layer(256, 128)# 64
        self.layer6 = self.make_layer(128, 64)# 128
        # self.layer7 = self.make_layer(128, 64)# 256
        self.layer7 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True),  nn.Tanh())
            # state size. ``(nc) x 64 x 64``

    def make_layer(self, in_dim, out_dim):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(True))
        return layer
        
    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # x = self.layer7(x)
        x = self.layer7(x)
        return x
    
class Discriminator_(nn.Module):
    def __init__(self):
        super(Discriminator_, self).__init__()
        # input is ``(nc) x 64 x 64``)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=True),   nn.LeakyReLU(0.2, inplace=True)) # 512
            # state size. ``(ndf) x 32 x 32``
        self.layer2 = self.make_layer(64, 64) # 256
        self.layer3 = self.make_layer(64, 128) # 128
        self.layer4 = self.make_layer(128, 128)#64
        self.layer5 = self.make_layer(128, 256)# 32
        self.layer6 = self.make_layer(256, 256)#16
        # self.layer7 = self.make_layer(256, 512)#8
        # self.layer8 = self.make_layer(512, 512)#4
        self.layer7 = nn.Sequential(
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(256, 1, 4, 1, 0, bias=True),
        )
    
    def make_layer(self, in_dim, out_dim):
        layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            )
        return layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # x = self.layer8(x)
        # x = self.layer9(x)
        x = x.view(x.size(0), -1)
        return x
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # discriminator = ResNet50_()
    discriminator = Discriminator_()
    generator = Generator()
    
    weights_init(discriminator)
    weights_init(generator)
    
    dataset = CustomImageDataset(
        # data_dir="/home/kangnam/datasets/images/cocosets/train2017/",
        # data_dir="/home/kangnam/datasets/images/kangnam_resize/",
        data_dir="/home/kangnam/datasets/images/sakimichan/",
        return_label=False,
        resolution=(256, 256),
        # data_length=1,
        data_repeat=1
    )
    # transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.ImageNet(root='./data/imagenet/train', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #                                         shuffle=True, num_workers=2)

    # testset = torchvision.datasets.ImageNet(root='./data/imagenet/valid', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=32,
    #                                        shuffle=False, num_workers=2)
    pipeline = GANPipeline(
        discriminator=discriminator, generator=generator, lr_d=5e-4, lr_g=5e-4, n_critic=1
    )

    # print(dataset.__len__())
    dataloder = DataLoader(dataset, batch_size=4)
    trainer = pl.Trainer(max_epochs=100000000)
    trainer.fit(model=pipeline, train_dataloaders=dataloder)


if __name__ == "__main__":
    main()
