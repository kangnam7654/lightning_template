import random
from pathlib import Path
from typing import Union

import cv2
import torch
from torchvision.transforms.functional import to_tensor

from datamodules.base import BaseDatamodule


class CustomImageDataset(BaseDatamodule):
    def __init__(
        self,
        data_dir: str,
        resolution=(512, 512),
        return_label=True,
        transform=None,
        z_norm=False,
        image_suffix=None,
        label_suffix=None,
        data_length=None,
        data_repeat=1,
    ):
        """
        Args:
            data_dirs (str or list, tuple): 이미지 파일이 저장된 디렉토리 경로.
            transform (callable, optional): 적용할 전처리 변환(Transforms).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_suffixes = self._parse_suffixes(
            image_suffix, self.image_default_suffixes
        )
        self.label_suffixes = self._parse_suffixes(
            label_suffix, self.label_default_suffixes
        )
        self.data_files = self.glob_files(self.data_dir, suffixes=self.image_suffixes)
        self.data_files = self._adjust_data_length(
            data_files=self.data_files,
            data_length=data_length,
            data_repeat=data_repeat,
        )

        self.resolution = resolution
        self._resolution_convert()
        self.z_norm = z_norm
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image_path = self.data_files[idx]
        image = self.image_process(str(image_path), resolution=self.resolution)
        if self.transform is not None:
            image = self.transform(image)

        out = []
        out.append(image)
        if self.return_label:
            label_path = self.get_label_path(
                data_path=image_path, label_suffixes=self.label_suffixes
            )
            label = self.label_process(label_path)
            label = label.float()
            out.append(label)
        return out

    def _resolution_convert(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            pass  # DO NOTHING

        else:
            raise ValueError(f"resolution을 확인하세요. 현재값 : {self.resolution}")


