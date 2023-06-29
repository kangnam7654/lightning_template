from pathlib import Path

import torch
import torch.nn.functional as F
from base import BasePipeline


class GANPipeline(BasePipeline):
    def __init__(
        self,
        generator=None,
        discriminator=None,
        manual_ckpt_save_path=None,
        lr_g=1e-2,
        lr_d=1e-2,
        n_critic=2,
        image_logging_interval=10,
    ):
        super().__init__()
        # 모델
        self.generator = generator
        self.discriminator = discriminator

        # 학습 인자
        self.automatic_optimization = False
        self.lr_d = lr_d
        self.lr_g = lr_g
        (
            self.generator_optimizer,
            self.discriminator_optimizer,
        ) = self.configure_optimizers()
        self.n_critic = n_critic
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

    def manual_save_checkpoint(self):
        state_dict = self.state_dict()
        current_save_path = self.rename_ckpt_path()
        torch.save(state_dict, current_save_path.resolve())
        if self._best_saved is not None:
            if self._best_saved.is_file():
                self._best_saved.unlink()
            else:
                print("%s은 파일이 아닙니다.", self._best_saved.resolve())
        self._best_saved = current_save_path
        print(f"\n{self._best_saved.resolve()} 저장 됨 \n")

    def save_last_epoch_checkpoint(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self._last_saved.resolve())

    def rename_ckpt_path(self, for_last=False):
        name = Path(self.manual_ckpt_save_path)
        if not for_last:
            to_add = f"epoch_{self.trainer.current_epoch}"
        else:
            name.parent.mkdir(parents=True, exist_ok=True)
            to_add = "last"
        renamed = name.with_stem(f"{name.stem}_{to_add}")
        return renamed

    def forward(self, z):
        image = self.generator(z)
        return image

    def training_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]
        view_shape = [bs, 1]
        latent_shape = [bs, 39, 1, 1]

        # Latent Vectors
        z = self.gaussian_sampling(latent_shape, sigma=1, type_as=image)
        label = label.type_as(image)

        # ==== Discriminator ====
        for _ in range(self.n_critic):
            self._discriminator_loop(image, z, view_shape)
            # self._discriminator_loop(image, label, view_shape)

        # ==== Generator =====
        gen_z = self.generator_loop(image, z, view_shape)
        gen_label = self.generator_loop(image, label, view_shape, is_label=True)

        # WANDB 이미지 로깅
        if self._image_logging_counter % self.image_logging_interval == 0:
            logger = self.logger.experiment
            self.logging_wandb_image(image, gen_z, gen_label, wandb_logger=logger)
            self._image_logging_counter = 1
        else:
            self._image_logging_counter += 1
        if self._show:
            self.image_show(image, gen_z, gen_label)

    def validation_step(self, batch, batch_idx):
        image, label = batch
        bs = image.shape[0]

        # 잠재 벡터 샘플
        z = self.gaussian_sampling([bs, 39, 1, 1], type_as=image)  # Latent Vector
        label = label.type_as(image)

        fake_image = self(z)
        fake_label_image = self(label)

        if self._show:
            self.image_show(image, fake_label_image, fake_image)
        fid_score = calculate_fid(
            real_images=image, generated_images=fake_label_image, batch_size=bs
        )

        # 평균 fid 점수 구함
        self._compute_average_fid_score(fid_score)

        # validation 로깅
        self.log("validation_fid", fid_score)

    def on_validation_epoch_end(self):
        self.log("validation_avg_fid", self._average_fid_score, prog_bar=True)
        if self._average_fid_score < self._highest_metric_score:
            self._highest_metric_score = self._average_fid_score
            self.manual_save_checkpoint()
        self._reset_fid_values()
        self.save_last_epoch_checkpoint()

    def _compute_average_fid_score(self, fid_socre):
        self._total_fid_score += fid_socre
        self._valid_step_counter += 1
        self._average_fid_score = self._total_fid_score / self._valid_step_counter

    def _reset_fid_values(self):
        self._total_fid_score = 0
        self._average_fid_score = 0
        self._valid_step_counter = 0

    def generator_loop(self, image, z, view_shape, is_label=False):
        # 이미지 생성
        fake_image = self.generator(z)
        fake_image_prediction = self.discriminator(fake_image)
        fake_image_prediction = fake_image_prediction.view(view_shape)

        real_image_prediction = self.discriminator(image)
        real_image_prediction = real_image_prediction.view(view_shape)

        # HARD 라벨
        real_labels = torch.ones(view_shape)
        real_labels = real_labels.type_as(image)  #

        loss = torch.mean((fake_image_prediction - 1) ** 2)
        self.log("generator_loss", loss, prog_bar=True)
        # # 잠재벡터 == 라벨
        # if is_label:
        #     # 손실함수 계산
        #     distribution_loss = self._compute_distribution_loss(
        #         fake_image_prediction, real_labels
        #     )
        #     reconstruction_loss = self._compute_reconstruction_loss(fake_image, image)
        #     loss = distribution_loss + reconstruction_loss

        #     # 로깅
        #     self.log("reconsruction_loss", loss, prog_bar=True)

        # # 잠재벡터 != 라벨
        # else:
        #     # 분포 손실 계산
        #     loss = self._compute_distribution_loss(fake_image_prediction, real_labels)
        #     # 로깅

        # 역전파
        self.generator_optimizer.zero_grad()
        self.manual_backward(loss)
        self.generator_optimizer.step()
        return fake_image

    def _discriminator_loop(self, image, z, view_shape):
        # 기울기 초기화
        self.discriminator_optimizer.zero_grad()

        # 라벨 생성
        real_label = self._get_soften_label(view_shape, True, image)
        fake_label = self._get_soften_label(view_shape, False, image)

        # FAKE 처리
        fake_image = self(z)
        fake_image_prediction = self.discriminator(fake_image.detach())
        fake_image_prediction = fake_image_prediction.view(view_shape)

        # REAL 처리
        real_image_prediction = self.discriminator(image)
        real_image_prediction = real_image_prediction.view(view_shape)

        # 손실함수 계산

        loss = torch.mean((real_image_prediction - 1) ** 2) + torch.mean(
            (fake_image_prediction) ** 2
        )

        # fake_loss = self._compute_distribution_loss(fake_image_prediction, fake_label)
        # real_loss = self._compute_distribution_loss(real_image_prediction, real_label)
        # loss = fake_loss + real_loss

        # Logging
        self.log("discriminator_loss", loss, prog_bar=True)

        # 역전파
        self.manual_backward(loss)
        self.discriminator_optimizer.step()
        self.untoggle_optimizer(self.discriminator_optimizer)

    def _get_soften_label(self, shape, real_labels=True, type_as=None):
        noise = torch.rand(shape) * 0.4
        if real_labels:
            labels = torch.ones(shape)
            labels = labels - noise
        else:
            labels = torch.zeros(shape)
            labels = labels + noise
        if type_as is not None:
            labels = torch.abs(labels)
            labels = labels.type_as(type_as)
        return labels

    def _compute_reconstruction_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def _compute_distribution_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999)
        )
        return generator_optimizer, discriminator_optimizer

    def gaussian_sampling(self, shape, type_as=None):
        z = torch.randn(shape)
        if type_as is not None:
            z = z.type_as(type_as)
        return z
