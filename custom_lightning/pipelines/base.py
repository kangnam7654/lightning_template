from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid

import wandb


class BasePipeline(pl.LightningModule):
    def __init__(self, manual_ckpt_save_path=None):
        super().__init__()
        # metrics
        self.total_train_loss = None

        self.total_valid_loss = None
        self.valid_step_counter = None
        self.manual_ckpt_save_path = manual_ckpt_save_path
        self.manual_best = None

    def _he_init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)  # inplace
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.weight.data.fill_(1)  # inplace
            m.bias.data.fill_(0)  # inplace

    def _xavier_init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    def apply_he_init(self, model):
        model.apply(self._he_init_weights)

    def apply_xavier_init(self, model):
        model.apply(self._xavier_init_weights)

    def invert_image_process(
        self, tensor: torch.Tensor, convert_bgr: bool = True
    ) -> np.uint8:
        """
        prediction 결과, Tanh 를 거쳐 (-1, 1)로 나온 결과를 다시 이미지로 되돌리는 함수입니다.
        """
        tensor = tensor.clone()

        if len(tensor.shape) == 4:
            images = []
            for t in tensor:
                tensor = t.permute(1, 2, 0)
                image = tensor.detach().cpu().numpy()
                image = np.clip(image, -1, 1)  # Assert
                image = (image + 1) * 127.5  # (-1, 1) -> (0, 255)
                image = image.astype(np.uint8)
                if convert_bgr:  # To cv2 order
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                images.append(image)
            images = np.stack(images, axis=0)
            return images

        elif len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)
            image = tensor.detach().cpu().numpy()
            image = np.clip(image, -1, 1)  # Assert
            image = (image + 1) * 127.5  # (-1, 1) -> (0, 255)
            image = image.astype(np.uint8)
            if convert_bgr:  # To cv2 order
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        else:
            raise ValueError(f"Shape를 확인하세요. 현재 shape : {tensor.shape}")

    def logging_wandb_image(
        self, *image_args, wandb_logger, no_convert_indieces=None, text="image"
    ):
        images = self.concat_tensor_images(
            *image_args, no_convert_indices=no_convert_indieces
        )
        wandb_logger.log({text: wandb.Image(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))})

    def concat_tensor_images(
        self,
        *tensor_images: torch.Tensor,
        no_convert_indices: Optional[List[int]] = [-1],
    ) -> np.ndarray:
        """
        이미지들을 이어 붙입니다.

        Args:
            tensor_images: reconstructed 이미지 입력입니다. 입력 순서대로 이어붙입니다.
            no_convert_indices: BGR 변환을 건너뛸 이미지의 인덱스를 리스트로 입력받습니다.
        """
        images = list(tensor_images)  # tensor_images == tuple
        nrow = len(images)

        # None 일경우 처리
        if no_convert_indices is None:
            no_convert_indices = []

        # -1은 마지막 인덱스 추가
        elif -1 in no_convert_indices:
            no_convert_indices.append(nrow)

        # 사진 샘플
        sample = images[0]
        aligned = []

        # 이미지 정렬
        for idx in range(len(sample)):
            for image_idx, image in enumerate(images):
                if image_idx in no_convert_indices:
                    aligned.append(image[idx])
                else:
                    aligned.append(image[idx][[2, 1, 0], :, :])  # RGB to BGR 변환

        grid = (
            make_grid(
                aligned,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1),
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return grid

    def image_show(self, *args, no_convert_indices=[-1]):
        image = self.concat_tensor_images(*args, no_convert_indices=no_convert_indices)
        cv2.imshow("concatenated image", image)
        cv2.waitKey(1)

    def manual_save_checkpoint(self, manual_ckpt_name, step, prefix="epoch"):
        """
        Pytorch Lightning의 Callback 이 정상적으로 작동한다면 필요없으나, 수동 역전파의 경우 필요함.
        """
        if manual_ckpt_name is None:
            print("checkpoint 경로가 설정되지 않았습니다.")
        else:
            # 새 체크포인트 저장
            state_dict = self.state_dict()  # 파이프라인 전체 statedict
            current_save_path = self.update_path(
                manual_ckpt_name, step, prefix
            )  # 이름 업데이트
            torch.save(state_dict, current_save_path.resolve())  # 저장
            print(f"{current_save_path}이 저장되었습니다.")

            # 이전 최고 체크포인트 삭제
            if self.manual_best is not None:  # 이전 저장 파일이 있을경우
                self.manual_best = Path(self.manual_best)

                if self.manual_best.is_file():
                    print(f"{self.manual_best}파일을 삭제합니다.")
                    self.manual_best.unlink()  # 이전 파일 삭제
                else:
                    print(f"{self.manual_best.resolve()}은 파일이 아닙니다.")

            # 최고 체크포인트 갱신
            self.manual_best = current_save_path

    def save_last_epoch_checkpoint(self, manual_last_path):
        if manual_last_path is not None:
            state_dict = self.state_dict()
            torch.save(state_dict, manual_last_path)
            print("마지막 파일을 저장합니다.")

    def update_path(
        self, manual_ckpt_save_path, step, prefix="epoch", for_last: bool = False
    ) -> Path:
        """
        수동 Checkpoint를 저장할 경우 사용 되는 메서드.
        체크포인트 저장 경로의 이름을 갱신 함

        Args:
            for_last (bool, optional): 마지막 Epoch용 이름. Defaults to False.

        Returns:
            pathlib.Path : 바뀐 이름
        """
        if manual_ckpt_save_path is not None:
            save_path = Path(manual_ckpt_save_path)

            # 디렉터리 없을 경우 생성
            if not save_path.parent.is_dir():
                save_path.parent.mkdir(parents=True, exist_ok=True)

            if not for_last:  # 일반적 상황
                to_add = f"{prefix}_{step}"
                renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")
            else:  # 무조건 저장
                to_add = "last"
                renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")

                ver = 1
                while renamed.is_file():
                    to_add = "last_v{}".format(ver)
                    renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")
                    ver += 1
            return renamed

    def _compute_epoch_avg_loss(
        self, is_training: bool = True, reset_args: bool = True
    ):
        if is_training:  # 학습
            avg = self.total_train_loss / self.train_step_counter
            if reset_args:
                self.total_train_loss = 0
                self.train_step_counter = 0
        else:
            avg = self.total_valid_loss / self.valid_step_counter
            if reset_args:
                self.total_valid_loss = 0
                self.valid_step_counter = 0
        return avg

    def update_parameters_with_ema(self, model_target, model_source, decay=0.999):
        """
        Update the parameters of the target model with the Exponential Moving Average (EMA)
        of the source model's parameters.

        Args:
        - model_target (nn.Module): The model with parameters to be updated.
        - model_source (nn.Module): The model providing the parameters for EMA calculation.
        - decay (float, optional): The decay rate for EMA. Default: 0.999.
        """

        target_params = dict(model_target.named_parameters())
        source_params = dict(model_source.named_parameters())

        for param_name in target_params.keys():
            target_param = target_params[param_name]
            source_param = source_params[param_name]

            # Update the target parameter with EMA
            target_param.data = (
                decay * target_param.data + (1 - decay) * source_param.data
            )

        # No need to return anything as we're updating model_target in-place

    def make_accumulate_dict(self, model):
        accumulate_dict = {name: torch.zeros_like(param.cuda()) for name, param in model.named_parameters()}
        return accumulate_dict

    def add_accumulate_dict(self, model, accumulate_dict):
        for name, param in model.named_parameters():
            accumulate_dict[name] += param.grad.clone()

    def apply_accumulate_dict(self, model, accumulate_dict, accumulate_interval):
        for name, param in model.named_parameters():
            param.grad = accumulate_dict[name] / accumulate_interval
            accumulate_dict[name].zero_()
