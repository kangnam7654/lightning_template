from typing import Optional, Union
from pathlib import Path
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import wandb


from utils.restore_image import invert_image_process


class BasePipeline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # metrics
        self.total_train_loss = None
        self.train_step_counter = None

        self.total_valid_loss = None
        self.valid_step_counter = None
        self.manual_save_ckpt_path = None
        self.manual_best = None
        self.manual_last = None

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

    def tensor_to_image(
        self, tensor: torch.Tensor, convert_bgr: bool = True
    ) -> np.uint8:
        return invert_image_process(tensor, convert_bgr=convert_bgr)

    def logging_wandb_image(self, *image_args, wandb_logger):
        images = self.concat_image(*image_args, convert_bgr=False)
        wandb_logger.log({"image": wandb.Image(images)})

    def concat_image(
        self,
        *tensor_images: torch.Tensor,
        convert_bgr: bool = True,
        horizontal: bool = True,
    ) -> np.ndarray:
        """
        Reconstruct 된 이미지들을 이어 붙입니다.

        Args:
            tensor_images: reconstructed 이미지 입력입니다. 입력 순서대로 이어붙입니다.
            convert_bgr: 모델의 결과 Tensor들은 RGB의 순서를 가집니다. 이를 opencv의 순서인 BGR로 변환합니다.
        """
        images = [
            self.tensor_to_image(x[0], convert_bgr=convert_bgr)
            for x in [*tensor_images]
        ]
        if horizontal:
            images = np.hstack(images)
        else:
            images = np.vstack(images)
        return images

    def image_show(self, *args):
        image = self.concat_image(*args)
        cv2.imshow("concatenated image", image)
        cv2.waitKey(1)

    def manual_save_checkpoint(self):
        """
        Pytorch Lightning의 Callback 이 정상적으로 작동한다면 필요없으나, 수동 역전파의 경우 필요함.
        """

        state_dict = self.state_dict()  # 파이프라인 전체 statedict
        current_save_path = self.rename_ckpt_path()  # 설정 이름 + epoch
        torch.save(state_dict, current_save_path.resolve())  # 저장

        if self.manual_best is not None:  # 이전 저장 파일이 있을경우
            self.manual_best = Path(self.manual_best)
            if self.manual_best.is_file():
                self.manual_best.unlink()  # 이전 파일 삭제
            else:
                print("%s은 파일이 아닙니다.", self.manual_best.resolve())

        self.manual_best = current_save_path
        print(f"\n{self.manual_best.resolve()} 저장 됨 \n")

    def save_last_epoch_checkpoint(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.manual_last.resolve())
        print("last checkpoint saved!")

    def rename_ckpt_path(self, for_last: bool = False) -> Path:
        """
        수동 Checkpoint를 저장할 경우 사용 되는 메서드.
        체크포인트 저장 경로의 이름을 갱신 함

        Args:
            for_last (bool, optional): 마지막 Epoch용 이름. Defaults to False.

        Returns:
            pathlib.Path : 바뀐 이름
        """
        if self.manual_save_ckpt_path is not None:
            save_path = Path(self.manual_save_ckpt_path)
            _temp_ver = 1

            if not save_path.parent.is_dir():  # 디렉터리 없을 경우 생성
                save_path.parent.mkdir(parents=True, exist_ok=True)

            if not for_last:  # 일반적 상황
                to_add = f"epoch_{self.trainer.current_epoch}"
                renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")
            else:  # 무조건 저장
                to_add = "last"
                renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")

                while renamed.is_file():
                    to_add = "last_v{}".format(_temp_ver)
                    renamed = save_path.with_stem(f"{save_path.stem}_{to_add}")
                    _temp_ver += 1
            return renamed

    def _add_loss(self, loss, is_training: bool = True):
        if is_training:
            self.total_train_loss += loss
            self.train_step_counter += 1
        else:
            self.total_valid_loss += loss
            self.valid_step_counter += 1

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
