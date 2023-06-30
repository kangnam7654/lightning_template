from pipelines.gan import GANPipeline
from pipelines.auto_encoder import ImitatorPipeline
from models.feature_extractors import ResNet50FeatureExtractor, ResNet152FeatureExtractor
from models.decoder import Decoder
from datamodules.image_module import CustomImageDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def main():
    dataset = CustomImageDataset(
        data_dir="/Users/kangnam/project/lightning_template/data/sample/image/",
        data_length=1,
        data_repeat=10000,
        return_label=False
    )
    dataloder = DataLoader(dataset, batch_size=1)
    discriminator = ResNet50FeatureExtractor(do_eval=True)
    generator = Decoder()
    gan = GANPipeline(
        generator=generator,
        discriminator=discriminator,
        manual_ckpt_save_path="./gan.ckpt",
        lr_d=1e-3,
        lr_g=1e-3,
        n_critic=1,
        image_logging_interval=100,
    )
    trainer = pl.Trainer(min_epochs=100)
    trainer.fit(model=ae, train_dataloaders=dataloder)


if __name__ == "__main__":
    main()
