import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BasePipeline


class ImitatorPipeline(BasePipeline):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        image_logging_interval: int = 10,
        lr: float = None,
        show: bool = False,
    ):
        super().__init__()
        # 모델
        self.encoder = encoder
        self.decoder = decoder

        # 학습 인자
        self.lr = lr

        # 점수 관련
        self.total_train_loss = 0
        self.train_step_counter = 0
        self.total_valid_loss = 0
        self.valid_step_counter = 0

        self.highest_score = 100

        # 로깅 관련
        self.image_logging_interval = image_logging_interval
        self._image_logging_counter = 1
        self._show = show

        # 체크포인트 저장
        self._manual_best_saved = None
        # self._manual_last_saved = self.rename_ckpt_path(for_last=True)

    def forward(self, image):
        latent = self.encode(image)
        reconstructed = self.decode(latent)
        return reconstructed

    def encode(self, image):
        latent = self.encoder(image)
        return latent

    def decode(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1, 1, 1)
        return self.decoder(x)

    def _in_loop(self, batch, is_training=True):
        image = batch[0]
        reconstructed = self.forward(image)

        reconstruction_loss = self.compute_loss(reconstructed, image)

        # # 이미지 로깅
        # if is_training:
        #     if self._can_log_image():
        #         logger = self.logger.experiment
        #         self.logging_wandb_image(image, reconstructed, wandb_logger=logger)
        #         self._image_logging_counter = 1
        #     else:
        #         self._image_logging_counter += 1
        if self._show:
            self.image_show(image, reconstructed)

        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        train_loss = self._in_loop(batch, is_training=True)

        # ===== loss 로깅 시작 =====
        # self._add_loss(train_loss, is_training=True)

        to_log_dict = self.metrics_to_dict(train_loss)
        self.log_dict(to_log_dict, prog_bar=True)
        # ===== loss 로깅 끝 =====
        return train_loss

    def on_train_epoch_end(self):
        # avg_loss = self._compute_epoch_avg_loss(is_training=True, reset_args=True)
        # self.log("avg_train_loss", avg_loss, prog_bar=True)
        pass

    def validation_step(self, batch, batch_idx):
        valid_loss = self._in_loop(batch, False)

        # 로깅 시작
        # self._add_loss(valid_loss, is_training=False)
        to_log_dict = self.metrics_to_dict(valid_loss, is_training=False)
        self.log_dict(to_log_dict, prog_bar=False)
        return valid_loss

    def on_validation_epoch_end(self):
        # avg_loss = self._compute_epoch_avg_loss(is_training=False, reset_args=True)
        # self.log("avg_val_loss", avg_loss, prog_bar=True)

        # # 체크포인트 저장
        # if avg_loss < self.highest_score:
        #     self.highest_score = avg_loss
        #     self.manual_save_checkpoint()
        # self.save_last_epoch_checkpoint()
        pass

    def configure_optimizers(self):
        decoder_optimizer = torch.optim.Adam(
            # list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.decoder.parameters(),
            lr=self.lr,
        )
        return decoder_optimizer

    def metrics_to_dict(self, loss, is_training: bool = True) -> dict:
        if is_training:
            mode = "train"
        else:
            mode = "valid"

        to_return = {
            f"{mode}_loss": loss,
        }
        return to_return

    def image_save(self, *args):
        image = self.concat_image(*args)
        # Save image
        cv2.imwrite(
            f"{self._image_save_folder}/image_{self.trainer.global_step}.jpg", image
        )

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        y_hat = y_hat.view(y.shape)
        loss = F.mse_loss(y_hat, y)
        return loss

    def _can_log_image(self) -> bool:
        if self._image_logging_counter % self.image_logging_interval == 0:
            return True
        else:
            return False

    def image_show(self, *args):
        image = [self.tensor_to_image(arg) for arg in args]
        image = np.concatenate(image, axis=1)
        cv2.imshow("", image)
        cv2.waitKey(1)