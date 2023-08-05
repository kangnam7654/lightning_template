import random
from pathlib import Path
from typing import Union

from PIL import Image

from data_modules.base import BaseDatamodule


class CustomImageDataset(BaseDatamodule):
    def __init__(
        self,
        data_dir: str,
        resolution=(512, 512),
        return_label=True,
        transform=None,
        image_ext=None,
        label_ext=None,
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

        # get preset extensions if exts are None
        self.image_extensions = self._parse_extensions(
            image_ext, self.image_default_extensions
        )
        self.label_extensions = self._parse_extensions(
            label_ext, self.label_default_extensions
        )

        # adjust -> data_files[:data_length] * data_repeat
        self.data_files = self.glob_files(self.data_dir, suffixes=self.image_extensions)
        self.data_files = self._adjust_data_length(
            data_files=self.data_files,
            data_length=data_length,
            data_repeat=data_repeat,
        )

        self.resolution = resolution
        self._resolution_convert() # int -> tuple
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image_path = self.data_files[idx]
        if self.transform is None:
            image = self.image_process(
                str(image_path), resolution=self.resolution
            )  # preset process
        else:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

        if not self.return_label:  # return label == False
            return image

        else:  # return label == True
            label_path = self.get_label_path(
                data_path=image_path, label_suffixes=self.label_extensions
            )
            label = self.label_process(label_path)
            label = label.float()
            return image, label

    def _resolution_convert(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            pass  # DO NOTHING

        else:
            raise ValueError(f"resolution을 확인하세요. 현재값 : {self.resolution}")
