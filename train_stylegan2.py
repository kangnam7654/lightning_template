import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_modules.image_module import CustomImageDataset
from lib.stylegan2.training.networks import Discriminator, Generator
from pipelines.stylegan2 import StyleGAN2Pipeline
from data_modules.lit import LightningDataWrapper2

torch.set_float32_matmul_precision("high")
pl.seed_everything(2023)


def load_model(pretrained=True):
    generator = Generator(z_dim=100, c_dim=0, w_dim=512, img_resolution=256, img_channels=3)
    discriminator = Discriminator(c_dim=0, img_resolution=256, img_channels=3)
    return generator, discriminator


def main():
    # ===== 모델 Load ======
    generator, discriminator = load_model(pretrained=False)
    batch_size = 1
    valid_size = 0.1
    resolution = 256

    pipeline = StyleGAN2Pipeline(
        generator=generator,
        discriminator=discriminator,
        manual_ckpt_save_path="./checkpoints/styelgan2_v1.ckpt",
        lr_d=1e-3,
        lr_g=1e-3,
        image_logging_interval=50,
        latent_dim=100,
    )
    
    # ===== Logger 설정 =====
    wandb_logger = WandbLogger(name="gan_test1_finetune", project="gan")
    wandb_logger.watch(pipeline, log="all")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== Trainer 불러오기 =====
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor],
        max_epochs=1000000,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
    )

    # ===== Dataset 설정 및 래핑 =====
    dataset = CustomImageDataset(
        data_dir="data/kangnam_resize",
        return_label=False,
        resolution=resolution,
        data_length=None,
        data_repeat=1,
    )
    wrapped_module = LightningDataWrapper2(
        dataset=dataset, batch_size=batch_size, num_workers=8, valid_size=valid_size
    )

    # 달려!
    trainer.fit(model=pipeline, datamodule=wrapped_module)


if __name__ == "__main__":
    main()
