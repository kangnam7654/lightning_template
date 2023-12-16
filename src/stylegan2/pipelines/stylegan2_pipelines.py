from typing import Union
from pathlib import Path
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

from kangnam_packages.pipelines.base import BasePipeline


class AnimeGANPipeline(BasePipeline):
    def __init__(
        self,
        generator: nn.Module = None,
        discriminator: nn.Module = None,
        manual_ckpt_save_path: Union[str, Path, None] = None,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        image_logging_interval: int = 10,
        latent_dim=100,
        show=False,
        return_label=False,
    ):
        super().__init__()
        self._internal_device = "cuda" if torch.cuda.is_available() else "cpu"

        # 모델
        self.generator = generator
        self.discriminator = discriminator

        # 학습 인자
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.return_label = return_label
        (
            self.generator_optimizer,
            self.discriminator_optimizer,
        ) = self.configure_optimizers()
        self.image_logging_interval = image_logging_interval
        self.show = show
        self.latent_shape = [latent_dim]

        # ADaptive Augmentation (ADA) for stylegan2
        self.ada_augment = AdaptiveAugment(0.5, 500 * 1000, 256)
        self.ada_aug_p = 0

        # Lazy regularization
        self.g_regularize_interval = 4
        self.d_regularize_interval = 8
        self.pl_mean = torch.empty([]).to(self._intetnal_device)

        self.manual_ckpt_save_path = manual_ckpt_save_path
        # Manual backward setting
        self.automatic_optimization = False
        self.ckpt_save_interval = 100
        self.check_manual_opt_state()

    def forward(self, z, c=None):
        if c is None:
            pass
        return self.generator(z, c)

    def style_mix_forward(self, z, c):
        return self.generator.style_mix_run(z, c)  # custom layer

    def training_step(self, batch, batch_idx):
        if self.return_label:
            real_images, label = batch
        else:
            real_images = batch
            label = None
        bs = real_images.size(0)
        latent_shape = [bs] + self.latent_shape  # [bs, latent_shape]

        # | Generator |
        self.requires_grad(self.generator, True)
        self.requires_grad(self.discriminator, False)
        g_loss, fake_images = self._generator_loop(
            real_images, label=label, latent_shape=latent_shape
        )

        # | Discriminator |
        self.requires_grad(self.generator, False)
        self.requires_grad(self.discriminator, True)
        d_loss, real_images_aug = self._discriminator_loop(
            real_images, label, latent_shape
        )

        # Loss Logging
        to_log = {
            "G_loss": g_loss,
            "D_loss": d_loss,
        }

        self.log_dict(to_log, prog_bar=True)

        # WANDB Image Logging
        if self._is_possible(self.image_logging_interval, contain_initial=True):
            logger = self.logger.experiment
            self.logging_wandb_image(
                real_images, real_images_aug, fake_images, wandb_logger=logger
            )

        # Image show
        if self.show:
            self.image_show(real_images, real_images_aug, fake_images)

        if self._is_possible(self.ckpt_save_interval):
            self.manual_save_checkpoint(
                self.manual_ckpt_save_path,
                (self.training_step_counter + 1),
                prefix="step",
            )
        self.training_step_counter += 1
        return to_log

    def _generator_loop(self, real_images, latent_shape, label=None):
        z = self.gaussian_sampling(latent_shape, type_as=real_images)  # Noise Sampling

        if label is not None:
            label = label.type_as(real_images)  # Prepared Condition

        fake_images = self.forward(z, label)
        fake_images_aug, _ = augment(fake_images, self.ada_aug_p)
        fake_preds = self.discriminator(fake_images_aug, label)
        g_loss = F.softplus(-fake_preds).mean()

        self.generator.zero_grad()
        self.manual_backward(g_loss)
        self.generator_optimizer.step()

        # Path length regularization
        if self._is_possible(self.g_regularize_interval):
            z = self.gaussian_sampling(latent_shape, type_as=real_images)
            fake_images, latents = self.style_mix_forward(z, label)

            g_path_loss = self.path_length_regularization(fake_images, latents)
            g_path_loss = g_path_loss * self.g_regularize_interval
            self.log("g_path_loss", g_path_loss, prog_bar=True)

            self.generator.zero_grad()
            self.manual_backward(g_path_loss)
            self.generator_optimizer.step()
        return g_loss, fake_images

    def _discriminator_loop(self, real_images, label, latent_shape):
        # Sampling z
        z = self.gaussian_sampling(latent_shape, type_as=real_images)
        if label is not None:
            label = label.type_as(real_images)  # Prepared Condition

        # Fake images preds
        fake_images = self.forward(z, label)
        fake_images_aug, _ = augment(fake_images, self.ada_aug_p)
        fake_preds = self.discriminator(fake_images_aug, label)

        # Real images preds
        real_images_aug, _ = augment(real_images, self.ada_aug_p)
        real_preds = self.discriminator(real_images_aug, label)

        # adjust aug probability p
        self.ada_aug_p = self.ada_augment.tune(real_preds)

        # Compute loss
        d_loss = F.softplus(-real_preds).mean() + F.softplus(fake_preds).mean()

        # Backward
        self.discriminator.zero_grad()
        self.manual_backward(d_loss)
        self.discriminator_optimizer.step()

        # R1 Regularization
        if self._is_possible(self.d_regularize_interval):
            # do augmentation before discriminate
            real_images_aug, _ = augment(real_images, self.ada_aug_p)
            real_images_aug.requires_grad = True  # To compute manual gradient
            real_preds = self.discriminator(real_images_aug, label)

            r1_loss = self.r1_regularization(real_preds, real_images_aug)
            r1_loss = r1_loss * self.d_regularize_interval
            self.log("r1_loss", r1_loss, prog_bar=True)

            self.discriminator.zero_grad()
            self.manual_backward(r1_loss)
            self.discriminator_optimizer.step()
        return d_loss, real_images

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999)
        )
        return generator_optimizer, discriminator_optimizer

    def gaussian_sampling(self, shape, sigma=1, type_as=None):
        z = torch.randn(shape) * sigma
        if type_as is not None:
            z = z.type_as(type_as)
        return z

    def r1_regularization(self, real_preds, real_images, r1_weight=10.0):
        grad_real = autograd.grad(
            outputs=real_preds.sum(),
            inputs=real_images,
            create_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = grad_real.square().sum([1, 2, 3])
        r1_loss = ((r1_weight / 2) * grad_penalty).mean()
        return r1_loss

    def path_length_regularization(
        self, fake_images, latents, pl_decay=0.01, pl_weight=2.0
    ):
        noise = torch.randn_like(fake_images).type_as(fake_images) / np.sqrt(
            fake_images.size(2) * fake_images.size(3)
        )
        grad = autograd.grad(
            outputs=(fake_images * noise).sum(),
            inputs=latents,
            create_graph=True,
        )[0]
        pl_lengths = grad.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        pl_loss = (pl_weight * pl_penalty).mean()
        return pl_loss
