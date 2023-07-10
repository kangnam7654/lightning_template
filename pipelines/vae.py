import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipelines.base import BasePipeline


class VAEPipeline(BasePipeline):
    def __init__(self, encoder, decoder, in_dim=768, lr=None, checkpoint=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._in_dim = in_dim
        self.latent_dim = 39
        self.mu_layer = nn.Linear(self._in_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self._in_dim, self.latent_dim)
        self.lr = lr
        self.checkpoint = checkpoint
        self._image_save_folder = "./image_logging"
        self.total_train_loss = 0
        self.cnt = 0

        # self.example_input_array = torch.zeros(1, 39, 1, 1)
        os.makedirs(self._image_save_folder, exist_ok=True)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def decode(self, z):
        reconstructed_image = self.decoder(z)
        return reconstructed_image

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Gaussian Sample
        z = mu + eps * std
        return z

    def _load_checkpoint(self):
        if self.checkpoint:
            state_dict = torch.load(self.checkpoint)["state_dict"]
            self.load_state_dict(state_dict)
            print("모델 CHECKPOINT 불러오기 완료")

    def training_step(self, batch, batch_idx):
        loss, bce, kl_divergence = self.loop(batch, is_training=True)
        self.cnt += 1
        self.total_train_loss += loss

        avg_loss = self.total_train_loss / self.cnt
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_bce", bce, prog_bar=True)
        self.log("train_kl_divergence", kl_divergence, prog_bar=True)
        self.log("avg_train_loss", avg_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, bce, kl_divergence = self.loop(batch, is_training=False)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_bce", bce)
        self.log("val_kl_divergence", kl_divergence)
        return val_loss

    def on_train_epoch_end(self):
        self.cnt = 0
        self.total_train_loss = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loop(self, batch, is_training=True):
        image = batch
        reconstructed, mu, logvar = self.forward(image)
        loss = self.compute_criterion(reconstructed, image, mu, logvar)
        # self.image_show(image, reconstructed)
        if self.trainer.global_step % 100 == 0:
            self.image_save(image, reconstructed, self.trainer.global_step)
        return loss

    def compute_criterion(self, reconstructed, image, mu, logvar):
        image = F.sigmoid(image)
        reconstructed = F.sigmoid(reconstructed)
        bce = F.binary_cross_entropy(reconstructed, image, reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = bce + kl_divergence
        return loss, bce, kl_divergence

    def image_show(self, x, reconstructed):
        x_sample = self.tensor_to_image(x)
        reconstructed_sample = self.tensor_to_image(reconstructed)

        # Stack horizontally (width-wise)
        image = np.hstack((x_sample, reconstructed_sample))

        # Save image
        cv2.imshow("train_image", image)
        cv2.waitKey(1)

    def image_save(self, x, reconstructed, step):
        x_sample = self.tensor_to_image(x)
        reconstructed_sample = self.tensor_to_image(reconstructed)

        # Stack horizontally (width-wise)
        image = np.hstack((x_sample, reconstructed_sample))

        # Save image
        cv2.imwrite(f"{self._image_save_folder}/image_{step}.jpg", image)
