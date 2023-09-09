import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.perceptual_loss import VGG16FeatureExtractor
from torch import autograd

# try:
#     from .base import BasePipeline
# except ImportError as e:
#     print(e)
#     from mvface_packages.pipelines.base import BasePipeline
from mvface_packages.pipelines.base import BasePipeline


class AnimeGANPipeline(BasePipeline):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        w_adv,
        w_con,
        w_gra,
        w_col,
        manual_ckpt_save_path: str = None,
        image_logging_interval: int = 10,
        pretraining_epoch=2,
        lr: float = None,
        show: bool = False,
        accumulate_interval=4,
    ):
        super().__init__()
        # 모델
        self.generator = generator
        self.discriminator = discriminator
        self.g_accum = self.make_accumulate_dict(self.generator)
        self.d_accum = self.make_accumulate_dict(self.discriminator)
        self.accumulate_interval = accumulate_interval
        # 학습 인자
        self.lr = lr

        # 점수 관련
        self.total_train_loss = 0
        self.train_step_counter = 0
        self.total_valid_loss = 0
        self.valid_step_counter = 0

        self.highest_score = 100
        # Loss weight
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_gra = w_gra
        self.w_col = w_col
        self.pretraining_epoch = pretraining_epoch

        # 로깅 관련
        self.image_logging_interval = image_logging_interval
        self._image_logging_counter = 1
        self._show = show

        # 체크포인트 저장
        self.automatic_optimization = False
        self.manual_ckpt_save_path = manual_ckpt_save_path
        self.manual_last = self.update_path(
            manual_ckpt_save_path=manual_ckpt_save_path, step=None, for_last=True
        )
        self.perceptual_model = VGG16FeatureExtractor()
        self.training_step_counter = 0

    def forward(self, image):
        return self.generator(image)

    def encode(self, x: torch.Tensor):
        return self.generator.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.generator.decoder(z)

    def pretraining(self, batch, batch_idx):
        self.generator.requires_grad = True
        self.discriminator.requires_grad = False

        real_images, _ = batch
        gen_images = self.forward(real_images)
        g_opt, _ = self.optimizers()

        real_feature = self.get_feature(real_images)
        gen_feature = self.get_feature(gen_images)
        reconst_loss = F.l1_loss(real_feature.detach(), gen_feature)
        self.log("pre_reconst_loss", reconst_loss, prog_bar=True)

        g_opt.zero_grad()
        self.manual_backward(10 * reconst_loss)
        g_opt.step()

        if self._can_log_image():
            logger = self.logger.experiment
            self.logging_wandb_image(
                real_images,
                gen_images,
                wandb_logger=logger,
                no_convert_indieces=None,
                text="pretraining",
            )  # 이미지 로깅
            self._image_logging_counter = 1
        else:
            self._image_logging_counter += 1
        # ================
        # = Logging Ends =
        # ================

        # 학습중 이미지 보기 설정
        if self._show:
            self.image_show(real_images, gen_images, no_convert_indices=None)
        return gen_images

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch < self.pretraining_epoch:  # For first epoch
            self.pretraining(batch, batch_idx)

        else:
            real_images, anime_images = batch  # source, target, edge blurring
            edge_images = self.blur_tensor_image(anime_images)
            # edge_images = self.rgb_to_grayscale(edge_images)
            gray_images = self.rgb_to_grayscale(anime_images)

            g_opt, d_opt = self.optimizers()  # optimizers

            # ========================
            # = Discriminator Starts =
            # ========================

            # Model Freeze
            self.discriminator.requires_grad = True
            self.generator.requires_grad = False

            # Generate Images
            gen_images = self.forward(real_images)

            # Get Predictions
            gen_preds = self.discriminator(gen_images)
            anime_preds = self.discriminator(anime_images)
            gray_preds = self.discriminator(gray_images)
            edge_preds = self.discriminator(edge_images)

            # Compute Discriminator Adversarial Loss
            """
            Softplus는 GAN에서 BCE를 근사할 수 있는 Loss입니다.
            참고페이지
                https://velog.io/@nochesita/%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%B4%EB%A1%A0-Binary-Cross-Entropy%EC%99%80-Softplus
            """
            # d_adv_anime_loss = 0.5 * torch.mean((anime_preds - 1) ** 2)
            # d_adv_gen_loss = 0.5 * torch.mean((gen_preds) ** 2)
            # d_adv_gray_loss = 0.5 * torch.mean((gray_preds) ** 2)
            # d_adv_edge_loss = 0.5 * torch.mean((edge_preds) ** 2)

            d_adv_anime_loss = F.softplus(-anime_preds).mean()
            d_adv_gen_loss = F.softplus(gen_preds).mean()
            d_adv_gray_loss = F.softplus(gray_preds).mean()
            d_adv_edge_loss = F.softplus(edge_preds).mean()

            d_loss = (
                d_adv_anime_loss
                + d_adv_gen_loss
                + (0.1 * d_adv_gray_loss)
                + (d_adv_edge_loss)
            ) * self.w_adv

            d_to_log = {
                "d_anime": d_adv_anime_loss,
                "d_gen": d_adv_gen_loss,
                "d_gray": d_adv_gray_loss,
                "d_edge": d_adv_edge_loss,
                "d_loss": d_loss,
            }

            self.log_dict(d_to_log, prog_bar=True)

            # Backwards
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            self.add_accumulate_dict(self.discriminator, self.d_accum)

            if (self.training_step_counter + 1) % self.accumulate_interval == 0:
                self.apply_accumulate_dict(
                    self.discriminator, self.d_accum, self.accumulate_interval
                )
                d_opt.step()
            # ======================
            # = Discriminator Ends =
            # ======================

            # ====================
            # = Generator Starts =
            # ====================

            # Model Freeze
            self.discriminator.requires_grad = False
            self.generator.requires_grad = True

            # Generate Images
            gen_images = self.forward(real_images)
            gray_gen_images = self.forward(gray_images)

            # Get Predictions
            gen_preds = self.discriminator(gen_images)

            # Feature Extracting by VGG
            content_feature = self.get_feature(real_images)
            reconst_feature = self.get_feature(gen_images)
            gray_feature = self.get_feature(gray_gen_images)

            # Compute Content Losss
            content_loss = F.l1_loss(reconst_feature, content_feature)

            # Compute gray sacle loss
            gram_gray = self.gram_matrix(gray_feature)
            gram_reconst = self.gram_matrix(reconst_feature)
            gray_style_loss = F.l1_loss(gram_gray, gram_reconst)

            # Compute Adversarial Loss
            # g_adv_loss = 0.5 * torch.mean((gen_preds - 1) ** 2)
            g_adv_loss = F.softplus(-gen_preds).mean()

            # Compute Colour Loss
            colour_loss = self.compute_colour_loss(
                content=real_images, reconst=gen_images
            )
            # Total Generator Loss
            g_loss = (
                (self.w_adv * g_adv_loss)
                + (self.w_con * content_loss)
                + (self.w_gra * gray_style_loss)
                + (self.w_col * colour_loss)
            )

            # Backwards
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.add_accumulate_dict(self.generator, self.g_accum)

            if (self.training_step_counter + 1) % self.accumulate_interval == 0:
                self.apply_accumulate_dict(self.generator, self.g_accum, self.accumulate_interval)
                g_opt.step()

            self.training_step_counter += 1
            # ==================
            # = Generator Ends =
            # ==================

            # ==================
            # = Logging Starts =
            # ==================
            g_to_log = {
                "g_adv": g_adv_loss,
                "g_content": content_loss,
                "g_gray": gray_style_loss,
                "g_colour": colour_loss,
                "g_loss": g_loss,
            }
            self.log_dict(g_to_log, prog_bar=True)  # 로깅

            # Image Logging
            if self._can_log_image():
                logger = self.logger.experiment
                self.logging_wandb_image(
                    real_images,
                    gen_images,
                    anime_images,
                    gray_images,
                    edge_images,
                    wandb_logger=logger,
                    text="training_images",
                    no_convert_indieces=None,
                )  # 이미지 로깅
                self._image_logging_counter = 1
            else:
                self._image_logging_counter += 1
            # ================
            # = Logging Ends =
            # ================

            # 학습중 이미지 보기 설정
            if self._show:
                self.image_show(
                    real_images,
                    gen_images,
                    anime_images,
                    gray_images,
                    edge_images,
                    no_convert_indices=None,
                )

    def on_train_epoch_end(self) -> None:
        self.save_last_epoch_checkpoint(manual_last_path=self.manual_last)

    # def validation_step(self, batch, batch_idx):
    #     valid_loss = self.loop(batch, False)
    #     to_log_dict = {"valid_loss": valid_loss}
    #     self.log_dict(to_log_dict, prog_bar=False)
    #     return valid_loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            betas=(0.5, 0.99),
            lr=self.lr,
        )
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(), betas=(0.5, 0.999), lr=self.lr
        )
        return g_opt, d_opt

    def _can_log_image(self) -> bool:
        if self._image_logging_counter % self.image_logging_interval == 0:
            return True
        else:
            return False

    def r1_regularization(self, real_preds, real_images, r1_weight=10.0):
        grad_real = autograd.grad(
            outputs=real_preds.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = grad_real.square().sum([1, 2, 3])
        r1_loss = ((r1_weight / 2) * grad_penalty).mean()
        return r1_loss

    def vgg_norm(self, x):
        x_ = (x.clone() + 1) / 2
        return x_

    def rgb_to_grayscale(self, images):
        # 이미지 텐서의 크기: (batch_size, 3, height, width)
        r, g, b = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]

        # 가중 평균을 사용하여 그레이스케일로 변환
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # 그레이스케일 이미지를 3채널로 복제
        grayscale_3channel = torch.cat([grayscale, grayscale, grayscale], dim=1)

        return grayscale_3channel

    def blur_tensor_image(self, tensor_img):
        # 배치 크기와 이미지 차원을 얻기
        batch_size, C, H, W = tensor_img.shape

        # 결과를 저장하기 위한 빈 텐서 초기화
        blurred_tensor_imgs = torch.zeros_like(tensor_img)

        for b in range(batch_size):
            # 한 이미지를 [0, 1] 범위로 역정규화
            img = ((tensor_img[b] + 1.0) / 2.0).clamp(0, 1)

            # CHW -> HWC, [0, 1] -> [0, 255]
            img = (img.permute(1, 2, 0) * 255.0).clone().detach().cpu().byte().numpy()

            # 1. Canny Edge Detection
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            # 2. Edge Dilation
            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # 3. Gaussian Smoothing in the dilated edge regions
            for i in range(3):  # RGB 각 채널에 대해
                img_channel = img[:, :, i]
                img_channel_blurred = cv2.GaussianBlur(img_channel, (5, 5), sigmaX=1)
                img[:, :, i][dilated_edges > 0] = img_channel_blurred[dilated_edges > 0]

            # 이미지를 텐서로 변환 및 정규화
            curr_tensor_img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            curr_tensor_img = curr_tensor_img * 2.0 - 1.0

            blurred_tensor_imgs[b] = curr_tensor_img

        return blurred_tensor_imgs

    def get_feature(self, x):
        x_ = self.vgg_norm(x)
        feature = self.perceptual_model(x_)
        return feature

    def gram_matrix(self, feature):
        """
        Computes the Gram matrix of a given feature.
        :param feature: torch.Tensor, feature matrix of shape (B, C, H, W)
        :return: torch.Tensor, Gram matrix of shape (B, C, C)
        """
        (b, c, h, w) = feature.size()
        feature = feature.view(b, c, h * w)
        feature_t = feature.transpose(1, 2)
        gram = torch.bmm(feature, feature_t)  # Batch matrix multiplication
        return gram / (c * h * w)  # Normalize by dividing by the number of elements

    def rgb_to_yuv(self, rgb_image):
        """
        Convert an RGB image to YUV
        :param rgb_image: torch.Tensor of shape (B, 3, H, W) with values in range [-1, 1]
        :return: torch.Tensor of shape (B, 3, H, W) with YUV values
        """

        # Convert from [-1, 1] to [0, 255]
        rgb_image = ((rgb_image + 1.0) * 127.5).clamp(0, 255)

        # Define the RGB to YUV transformation matrix
        # These coefficients are typically used for the YUV conversion
        transform = (
            torch.tensor(
                [
                    [0.299, -0.14714119, 0.61497538],
                    [0.587, -0.28886916, -0.51496512],
                    [0.114, 0.43601035, -0.10001026],
                ]
            )
            .t()
            .to(rgb_image.device)
        )

        # Apply the transformation
        yuv_image = torch.matmul(
            transform, rgb_image.view(-1, 3, rgb_image.shape[2] * rgb_image.shape[3])
        )
        yuv_image = yuv_image.view(rgb_image.shape)

        # Convert YUV values back to [-1, 1] range for consistency
        # Note: This step is optional and depends on your needs
        yuv_image = (yuv_image / 127.5) - 1.0

        return yuv_image

    def compute_colour_loss(self, content, reconst):
        content_yuv = self.rgb_to_yuv(content)
        reconst_yuv = self.rgb_to_yuv(reconst)

        content_y, content_u, content_v = torch.split(content_yuv, 1, dim=1)
        reconst_y, reconst_u, reconst_v = torch.split(reconst_yuv, 1, dim=1)
        y_loss = F.l1_loss(reconst_y, content_y)
        u_loss = F.huber_loss(reconst_u, content_u)
        v_loss = F.huber_loss(reconst_v, content_v)
        loss = y_loss + u_loss + v_loss
        return loss
