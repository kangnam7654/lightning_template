from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pipelines.metaworld_base import MetaWorldBasePipeline


class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        e1 = F.normalize(e1)
        e2 = F.normalize(e2)
        similarity = torch.cosine_similarity(e1, e2)
        # ones = torch.ones_like(similarity)
        loss = 1 - similarity.mean()
        return loss


class BatchDistributionLoss(nn.Module):
    """
    입력 : 캐릭터 캡처 + 라벨 + 인물 사진
    """

    def __init__(self, continuous_dim=36):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.criterion = nn.MSELoss()

    def forward(self, c1, d1, c2, d2):
        # c.shape == (bs, 36)
        # d.shape == (bs, n_class, 3)
        d1 = torch.softmax(d1, dim=1)
        d2 = torch.softmax(d2, dim=1)

        c1_m, c1_s = self.get_statistics(c1)
        c2_m, c2_s = self.get_statistics(c2)
        d1_m, d1_s = self.get_statistics(d1)
        d2_m, d2_s = self.get_statistics(d2)

        c_mu = self.criterion(c1_m, c2_m)
        c_sigma = self.criterion(c1_s, c2_s)
        d_mu = self.criterion(d1_m, d2_m)
        d_sigma = self.criterion(d1_s, d2_s)
        loss = c_mu + c_sigma + d_mu + d_sigma
        return loss

    def get_statistics(self, tensor, continuous=True):
        if continuous:
            mean = torch.mean(tensor, dim=1)
            var = torch.std(tensor, dim=1)
        return mean, var


class ParameterLoss(nn.Module):
    def __init__(self, w_c=1, w_d=1, continuous_dim=36):
        super().__init__()
        self.w_c = w_c
        self.w_d = w_d
        self.continuous_dim = continuous_dim
        self.continuous_criterion = nn.MSELoss()
        self.discrete_criterion = nn.CrossEntropyLoss()

    def forward(self, c, d, label):
        label_c = label[:, : self.continuous_dim]
        label_d = label[:, self.continuous_dim :]

        c_loss = self.continuous_criterion(c, label_c)
        d_loss = self.discrete_criterion(d, label_d.long())
        loss = self.w_c * c_loss + self.w_d * d_loss
        return loss


class LoopbackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_criterion = nn.MSELoss()
        self.d_criterion = nn.CrossEntropyLoss()

    def forward(self, c1, d1, c2, d2):
        d1 = torch.softmax(d1, dim=1)  # (bs, C, 3)
        d2 = torch.softmax(d2, dim=1)
        c_loss = self.c_criterion(c1, c2)
        d_loss1 = self.d_criterion(d1, d2)
        d_loss2 = self.d_criterion(d2, d1)
        d_loss = (d_loss1 + d_loss2) * 0.5
        loss = c_loss + d_loss
        return loss


class MainNetPipeline(MetaWorldBasePipeline):
    def __init__(
        self,
        recognizer=None,
        translator=None,
        imitator=None,
        segmenter=None,
        mapping_net=None,
        lr=None,
    ):
        super().__init__()
        self.recognizer = recognizer
        self.translator = translator
        self.imitator = imitator
        self.segmenter = segmenter  # output -> (batch, nc, h, w)
        self.mapping_net = mapping_net

        # loss functions
        self.identity_loss_function = IdentityLoss()
        self.parameter_loss_function = ParameterLoss()
        self.loopback_loss_function = LoopbackLoss()
        self.batch_dist_loss_function = BatchDistributionLoss()

        self.lr = lr

    def forward(self, image: torch.Tensor):
        recognized = self.recognize(image)
        continuous, discrete = self.translate(recognized)
        return recognized, continuous, discrete

    def recognize(self, image):
        return self.recognizer(image)

    def translate(self, recognized):
        continuous, discrete = self.translator(recognized)
        return continuous, discrete

    def imitate(self, translated):
        mapped = self.mapping(translated)
        mapped = mapped.view(mapped.shape[0], -1, 1, 1)
        return self.imitator(mapped)

    def mapping(self, translated):
        translated = torch.flatten(translated, 1)
        return self.mapping_net(translated)

    def segment(self, x):
        seg_map = self.segmenter(x)
        return seg_map

    def training_step(self, batch, batch_idx):
        loss = self.loop(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loop(batch, is_training=False)
        return loss

    def loop(self, batch, is_training=True):
        """
        where
        c = continuous
        d = discrete
        e = embedding
        w = whiten
        """
        real_image, unreal_image, label = batch  # face cropped 160

        # Whiten Image
        w_real_image = self.instance_z_norm(real_image)
        w_unreal_image = self.instance_z_norm(unreal_image)

        # batch recognize
        real_e = self.recognize(w_real_image)
        unreal_e = self.recognize(w_unreal_image)

        # real image style transfer
        real_e_adain = self.adain(real_e, unreal_e)

        # Parameter translation
        real_c, real_d = self.translate(real_e_adain)
        unreal_c, unreal_d = self.translate(unreal_e)
        real_pred = self.temp_refine(real_c, real_d)

        # 렌더링 (Rendering)
        imitated = self.imitate(real_pred)
        imitated = F.interpolate(
            imitated,
            size=[real_image.shape[2], real_image.shape[3]],
            mode="bilinear",
            align_corners=False,
        )  # 리사이즈 (Resize )

        w_imitated = self.batch_z_norm(imitated)
        imitated_e = self.recognize(w_imitated)
        imitated_e_adain = self.adain(imitated_e, unreal_e)
        imitated_c, imitate_d = self.translate(imitated_e_adain)

        # ===== 손실 계산 ====
        parameter_loss = self.compute_parameter_loss(unreal_c, unreal_d, label)
        identity_loss = self.compute_identity_loss(real_e_adain, imitated_e_adain)
        loopback_loss = self.compute_loopback_loss(
            real_c, real_d, imitated_c, imitate_d
        )
        real_batch_dist_loss = self.compute_batch_dist_loss(
            real_c, real_d, imitated_c, imitate_d
        )

        loss = (
            parameter_loss
            + (identity_loss * 1)
            + (loopback_loss * 0.01)
            + (real_batch_dist_loss * 5)
        )
        to_log_dict = self.metrics_to_dict(
            parameter_loss=parameter_loss,
            identity_loss=identity_loss,
            loopback_loss=loopback_loss,
            real_batch_dist_loss=real_batch_dist_loss,
            loss=loss,
            is_training=is_training,
        )
        self.log_dict(to_log_dict, prog_bar=True)
        self.image_show(real_image, imitated)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.translator.parameters(),
            lr=self.lr,
        )
        return optimizer

    def metrics_to_dict(
        self,
        parameter_loss,
        identity_loss,
        loopback_loss,
        real_batch_dist_loss,
        loss,
        is_training=True,
    ) -> dict:
        if is_training:
            mode = "train"
        else:
            mode = "valid"

        to_return_dict = {
            f"{mode}_parameter": parameter_loss,
            f"{mode}_identity": identity_loss,
            f"{mode}_loopback": loopback_loss,
            f"{mode}_batch_dist": real_batch_dist_loss,
            f"{mode}_loss": loss,
        }
        return to_return_dict

    def light_cnn_preprocess(
        self, image: torch.Tensor, target_size=(128, 128)
    ) -> torch.Tensor:
        # image (512, 512), range (-1, 1)
        image = (image + 1) * 0.5
        light_cnn_image = F.interpolate(
            image, size=target_size, mode="bilinear", align_corners=False
        )
        return light_cnn_image

    def raw_image_preprocess(
        self, image_path: Union[str, Path], target_size=(128, 128)
    ):
        image = cv2.imread(image_path, 0)
        image = image / 255
        image = torch.from_numpy(image)
        image = F.interpolate(
            image, size=target_size, mode="bilinear", align_corners=False
        )
        return image

    def compute_parameter_loss(self, c, d, label):
        loss = self.parameter_loss_function(c, d, label)
        return loss

    def compute_loopback_loss(self, c1, d1, c2, d2):
        loss = self.loopback_loss_function(c1, d1, c2, d2)
        return loss

    def compute_batch_dist_loss(self, c1, d1, c2, d2):
        loss = self.batch_dist_loss_function(c1, d1, c2, d2)
        return loss

    def compute_identity_loss(self, embedding1, embedding2):
        loss = self.identity_loss_function(embedding1, embedding2)
        return loss

    def temp_refine(self, c, d):
        # d.shape == [bs, C, 3]
        d = torch.softmax(d, dim=1)
        d = torch.argmax(d, dim=1)
        d = d.type_as(c)
        concated = torch.concat([c, d], dim=-1)
        return concated

    def batch_z_norm(self, x: torch.Tensor):
        mean, std = self.get_statistics(x)
        normalized = (x - mean) / std
        normalized.type_as(x)
        return normalized

    def instance_z_norm(self, x: torch.Tensor):
        mean, std = self.get_statistics(x, dim=[2, 3])
        normalized = (x - mean) / std
        normalized.type_as(x)
        return normalized

    def get_statistics(self, x, dim=None):
        mean = torch.mean(x, dim=dim, keepdim=True if dim is not None else False)
        std = torch.std(x, dim=dim, keepdim=True if dim is not None else False)

        eps = torch.tensor([1e-7])
        eps = eps.expand_as(eps)
        eps = eps.type_as(x)

        std_adj = torch.maximum(std, eps)
        return mean, std_adj

    def get_statistics_(self, x):
        eps = torch.tensor([1e-7])
        eps = eps.type_as(x)

        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        std_adj = torch.maximum(std, eps)
        return mean, std_adj

    def adain(self, real, character):
        real_mean, real_std = self.get_statistics_(real)
        character_mean, character_std = self.get_statistics_(character)

        whiten = (real - real_mean) / real_std

        s_transfered = whiten * character_std + character_mean
        return s_transfered
