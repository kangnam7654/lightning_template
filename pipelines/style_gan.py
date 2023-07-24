from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .base import BasePipeline


class StyleGanPipeline(BasePipeline):
    def __init__(
        self,
        generator=None,
        discriminator=None,
        mapper=None,
        manual_ckpt_save_path=None,
        lr_g=1e-2,
        lr_d=1e-2,
        n_critic=1,
        image_logging_interval=10,
    ):
        super().__init__()
        # 모델
        self.generator = generator
        self.discriminator = discriminator
        self.mapper = mapper

        # 학습 인자
        self.automatic_optimization = False
        self.lr_d = lr_d
        self.lr_g = lr_g
        (
            self.opt_g,
            self.opt_d,
        ) = self.configure_optimizers()
        self.n_critic = n_critic
        self.image_logging_interval = image_logging_interval

        # 체크포인트 저장
        self.manual_ckpt_save_path = manual_ckpt_save_path
        self._best_saved = None
        if self.manual_ckpt_save_path is not None:
            self._last_saved = self.rename_ckpt_path(for_last=True)

        self._image_logging_counter = 1
        self._highest_metric_score = 1e8
        self._total_fid_score = 0
        self._average_fid_score = 0
        self._valid_step_counter = 0
        self._show = True
        self.latent_shape = [100]
        self._constant = None

    def style_mapping(self, z):
        w = self.mapper(z)
        return w

    def make_constant(self, data):
        if self._constant is None:
            self._constant = torch.rand(data.size(0), 512, 4, 4)

    def make_noise(self, data):
        self._noise = torch.rand(data.size(0), 512, 1, 1)

    def forward(self, content, mapped, noise):
        image = self.generator(content, mapped, noise)
        return image

    def training_step(self, batch, batch_idx):
        image = batch[0]

        # assert isinstance(image, torch.Tensor)
        bs = image.size(0)
        latent_shape = [bs] + self.latent_shape

        # ==== Discriminator ====
        for _ in range(self.n_critic):
            d_loss, d_fake_image = self.discriminator_loop(image, latent_shape)

            # backward
            self.opt_d.zero_grad()
            self.manual_backward(d_loss)
            self.opt_d.step()

            d_to_log = {"d_loss": d_loss}
            self.log_dict(d_to_log, prog_bar=True)

        # ==== Generator =====
        g_loss, g_fake_image = self.generator_loop(image, latent_shape)

        # backward
        self.opt_g.zero_grad()
        self.manual_backward(g_loss)
        self.opt_g.step()

        g_to_log = {"g_loss": g_loss}
        self.log_dict(g_to_log, prog_bar=True)
        # self.untoggle_optimizer(self.opt_g)

        # # TODO WANDB 이미지 로깅
        # if self._image_logging_counter % self.image_logging_interval == 0:
        #     logger = self.logger.experiment
        #     self._image_logging_counter = 1
        # else:
        #     self._image_logging_counter += 1
        if self._show:
            self.image_show(image, g_fake_image, d_fake_image)

    def validation_step(self, batch, batch_idx):
        # image = batch
        # bs = image.size(0)
        # latent_size = list(bs).extend(self.latent_shape)

        # # 잠재 벡터 샘플
        # z = self.gaussian_sampling(latent_size, type_as=image)  # Latent Vector
        # fake_image = self.forward(z)

        # TODO ============== FID metic =============
        # fid_score = calculate_fid(
        #     real_images=image, generate=fake_label_image, batch_size=bs
        # )

        # 평균 fid 점수 구함
        # self._compute_average_fid_score(fid_score)

        # validation 로깅
        # self.log("validation_fid", fid_score)
        pass

    def on_validation_epoch_end(self):
        # self.log("validation_avg_fid", self._average_fid_score, prog_bar=True)
        # if self._average_fid_score < self._highest_metric_score:
        #     self._highest_metric_score = self._average_fid_score
        #     self.manual_save_checkpoint()
        # self._reset_fid_values()
        latent_shape = list(2).extend(self.latent_shape)
        z = self.gaussian_sampling(
            latent_shape, type_as=list(self.generator.children)[0].weight
        )
        sample_images = self.forward(z)
        grid = torchvision.utils.make_grid(sample_images)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        self.manual_save_checkpoint()
        self.save_last_epoch_checkpoint()

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999)
        )
        return generator_optimizer, discriminator_optimizer

    def compute_loss(self, prediction, label):
        return F.binary_cross_entropy(prediction, label)
        # loss = F.binary_cross_entropy_with_logits(prediction, label)
        # loss = 0.5 * torch.pow((prediction - label), 2)
        # loss = torch.mean(loss)
        # loss = F.l1_loss(prediction, label)
        # return loss

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

    def rename_ckpt_path(self, for_last: bool = False):
        name = Path(self.manual_ckpt_save_path)
        if not for_last:
            to_add = f"epoch_{self.trainer.current_epoch}"
        else:
            name.parent.mkdir(parents=True, exist_ok=True)
            to_add = "last"
        renamed = name.with_stem(f"{name.stem}_{to_add}")
        return renamed

    def metric_dict(self, g_loss, d_loss, is_train: bool = True, fid=None):
        if is_train:
            mode = "train"
        else:
            mode = "valid"
        to_log = {f"{mode}_g_loss": g_loss, f"{mode}_d_loss": d_loss}
        if fid:
            to_log[f"{mode}_fid"] = fid
        return to_log

    def metric(self):
        # TODO fid score 서ㄹ정
        pass

    def _compute_average_fid_score(self, fid_socre):
        self._total_fid_score += fid_socre
        self._valid_step_counter += 1
        self._average_fid_score = self._total_fid_score / self._valid_step_counter

    def _reset_fid_values(self):
        self._total_fid_score = 0
        self._average_fid_score = 0
        self._valid_step_counter = 0

    def generator_loop(self, real_images, latent_shape):
        # 이미지 생성
        bs = real_images.size(0)
        content = torch.rand(bs, 512, 4, 4)
        content = content.type_as(real_images)
        
        
        z = torch.rand(bs, 512)
        z = z.type_as(real_images)
        mapped = self.mapper(z)
        
        noise = torch.rand(bs, 512, 1, 1)
        noise = noise.type_as(real_images)
        
        fake_images = self.forward(content, mapped, noise)
        fake_image_prediction = self.discriminator(fake_images)

        loss = -torch.mean(fake_image_prediction)
        return loss, fake_images

    def discriminator_loop(self, image, latent_shape):
        # FAKE 처리
        bs = image.size(0)
        content = torch.zeros(bs, 512, 4, 4)
        content = content.type_as(image)
        
        z = torch.randn(bs, 512)
        z = z.type_as(image)
        mapped = self.mapper(z)
        
        noise = torch.randn(bs, 512, 1, 1)
        noise = noise.type_as(image)
        
        fake_image = self.forward(content, mapped, noise)
        fake_image_prediction = self.discriminator(fake_image)

        # REAL 처리
        real_image_prediction = self.discriminator(image)

        gp = self.gradient_penalty(self.discriminator, image, fake_image)
        d_loss = (
            torch.mean(fake_image_prediction) - torch.mean(real_image_prediction) + gp
        )
        return d_loss, fake_image

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

    def gaussian_sampling(self, shape, mean: float = 0, sigma: float = 1, type_as=None):
        z = torch.randn(shape)
        z = (z - mean) * sigma
        if type_as is not None:
            z = z.type_as(type_as)
        return z

    def image_show(self, *args):
        image = [self.temp_reconstruct(arg) for arg in args]
        image = np.concatenate(image, axis=1)
        cv2.imshow("", image)
        cv2.waitKey(1)

    def temp_reconstruct(self, image: torch.Tensor):
        # image = F.interpolate(image, 512)
        image = image.clone().detach().cpu().numpy()[0]
        image = (image + 1) * 127.5
        image = np.transpose(image, (1, 2, 0))  # (c, h, w) -> (h, w, c)
        image = image.astype(np.uint8)
        image = image[:, :, ::-1]
        return image

    def gradient_penalty(self, discriminator, real_images, fake_images):
        alpha = torch.randn(real_images.size(0), 1, 1, 1).type_as(real_images)
        interpolates = (
            alpha * real_images + ((1 - alpha) * fake_images)
        ).requires_grad_(True)
        interpolates
        validity = discriminator(interpolates)
        fake = torch.ones_like(validity)
        gradients = torch.autograd.grad(
            outputs=validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
