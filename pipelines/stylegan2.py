import math
from typing import Union
from pathlib import Path
import numpy as np
import torch
from torch import autograd, optim
import torch.nn as nn
import torch.nn.functional as F
from pipelines.base import BasePipeline

import wandb
from utils.inception_score import calculate_fid
from lib.stylegan2.non_leaking import AdaptiveAugment, augment


class StyleGAN2Pipeline(BasePipeline):
    def __init__(
        self,
        generator: nn.Module = None,
        discriminator: nn.Module = None,
        manual_ckpt_save_path: Union[str, Path, None] = None,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        image_logging_interval: int = 10,
        latent_dim=100,
    ):
        super().__init__()
        # 모델
        self.generator = generator
        self.discriminator = discriminator

        # 학습 인자
        self.automatic_optimization = False  # 중요!!!!!
        self.lr_d = lr_d
        self.lr_g = lr_g
        (
            self.generator_optimizer,
            self.discriminator_optimizer,
        ) = self.configure_optimizers()
        self.image_logging_interval = image_logging_interval

        # 체크포인트 저장
        self.manual_ckpt_save_path = manual_ckpt_save_path
        self._best_saved = None
        self._last_saved = self.rename_ckpt_path(for_last=True)

        self._image_logging_counter = 1
        self._highest_metric_score = 1e8
        self._total_fid_score = 0
        self._average_fid_score = 0
        self._valid_step_counter = 0
        self._show = True

        self.latent_shape = [latent_dim]
        self.ada_augment = AdaptiveAugment(0.5, 1e5, 512, "cuda")
        self.ada_aug_p = 0

        self.g_regularize_interval = 4
        self.g_regularize_step = 0
        self.d_regularize_interval = 8
        self.d_regularize_step = 0
        self.pl_mean = torch.empty([]).cuda()

    def forward(self, z, c):
        return self.generator(z, c)

    def style_mix_forward(self, z, c):
        return self.generator.style_mix_run(z, c)  # custom layer

    def training_step(self, batch, batch_idx):
        try:
            real_images, label = batch
        except:
            real_images = batch
            label = torch.empty(real_images.size(0), 1)
        bs = real_images.size(0)
        latent_shape = [bs] + self.latent_shape

        # ===== Generator =====
        self.requires_grad(self.generator, True)
        self.requires_grad(self.discriminator, False)
        g_loss, fake_images = self._generator_loop(real_images, label, latent_shape)

        # ===== Discriminator =====
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

        # WANDB 이미지 로깅
        if self._image_logging_counter % self.image_logging_interval == 0:
            logger = self.logger.experiment
            self.logging_wandb_image(
                real_images, real_images_aug, fake_images, wandb_logger=logger
            )
            self._image_logging_counter = 1
        else:
            self._image_logging_counter += 1
        if self._show:
            self.image_show(real_images, real_images_aug, fake_images)

    def validation_step(self, batch, batch_idx):
        try:
            real_images, label = batch
            label = label.type_as(real_images)
        except:
            real_images = batch
            label = torch.empty(real_images.size(0), 1)
        bs = real_images.size(0)
        latent_shape = [bs] + self.latent_shape

        # 잠재 벡터 샘플
        z = self.gaussian_sampling(latent_shape, type_as=real_images)  # Latent Vector
        fake_images = self.forward(z, label)
        fid_score = calculate_fid(
            real_images=real_images, generated_images=fake_images, batch_size=bs
        )

        # 평균 fid 점수 구함
        self._compute_average_fid_score(fid_score)

        # validation 로깅
        self.log("validation_fid", fid_score)

    def on_validation_epoch_end(self):
        self.log("validation_avg_fid", self._average_fid_score, prog_bar=True)
        if self._average_fid_score < self._highest_metric_score:
            self._highest_metric_score = self._average_fid_score
            # self.manual_save_checkpoint()
        self._reset_fid_values()
        # self.save_last_epoch_checkpoint()

    def _compute_average_fid_score(self, fid_socre):
        # TODO 이거 필요한지 검토
        self._total_fid_score += fid_socre
        self._valid_step_counter += 1
        self._average_fid_score = self._total_fid_score / self._valid_step_counter

    def _reset_fid_values(self):
        self._total_fid_score = 0
        self._average_fid_score = 0
        self._valid_step_counter = 0

    def _generator_loop(self, real_images, label, latent_shape):
        self.g_regularize_step += 1
        z = self.gaussian_sampling(latent_shape, type_as=real_images)  # Noise Sampling
        label = label.type_as(label)  # Prepared Condition

        fake_images = self.forward(z, label)
        fake_images_aug, _ = augment(fake_images, self.ada_aug_p)
        fake_preds = self.discriminator(fake_images_aug, label)
        g_loss = F.softplus(-fake_preds).mean()
        self.generator.zero_grad()
        self.manual_backward(g_loss)
        self.generator_optimizer.step()

        # Path length regularization
        if self.g_regularize_step % self.g_regularize_interval == 0:
            z = self.gaussian_sampling(latent_shape, type_as=real_images)
            fake_images, latents = self.style_mix_forward(z, label)

            g_path_loss = self.path_length_regularization(fake_images, latents)
            g_path_loss = g_path_loss * self.g_regularize_interval
            self.log("g_path_loss", g_path_loss, prog_bar=True)

            self.generator.zero_grad()
            self.manual_backward(g_path_loss)
            self.generator_optimizer.step()
            self.g_regularize_step = 0
        return g_loss, fake_images

    def _discriminator_loop(self, real_images, label, latent_shape):
        self.d_regularize_step += 1

        # Sampling z
        z = self.gaussian_sampling(latent_shape, type_as=real_images)
        label = label.type_as(real_images)

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
        if self.d_regularize_step % self.d_regularize_interval == 0:
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
            self.d_regularize_step = 0
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

    def requires_grad(self, model, flag=True):
        for param in model.parameters():
            param.requires_grad = flag

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
            outputs=(fake_images * noise).sum(), inputs=latents, create_graph=True
        )[0]
        pl_lengths = grad.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        pl_loss = (pl_weight * pl_penalty).mean()
        return pl_loss
