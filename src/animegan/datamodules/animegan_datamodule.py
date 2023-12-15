import random
import sys
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from kangnam_packages.datamodules.base import BaseDatamodule


class AnimeGANDatamodule(BaseDatamodule):
    def __init__(
        self,
        content_image_dir: Optional[str] = None,
        style_image_dir: Optional[str] = None,
        resolution: Union[int, list, tuple] = 256,
        image_extension: str = None,
        label_extension: str = None,
        data_length: Union[int, float, None] = None,
        data_repeat: int = 1,
    ):
        """
        Args:
            data_dirs (str or list, tuple): 이미지 파일이 저장된 디렉토리 경로.
            transform (callable, optional): 적용할 전처리 변환(Transforms).
        """
        super().__init__()
        self.content_image_dir = content_image_dir
        self.style_image_dir = style_image_dir

        self.resolution = self._convert_resolution_type_to_int(resolution)
        self.data_length = data_length
        self.data_repeat = data_repeat
        self.image_extensions = self._extensions_to_list(
            image_extension, self.image_default_extensions
        )
        self.label_extensions = self._extensions_to_list(
            label_extension, self.label_default_extensions
        )

        # ===== FILE LOAD ======
        self.content_image_files = self.glob_files(
            content_image_dir, self.image_extensions
        )
        self.style_image_files = self.glob_files(style_image_dir, self.image_extensions)

        if len(self.content_image_files) > 0:
            self.content_image_files = self._adjust_data_length(
                self.content_image_files, self.data_length, self.data_repeat
            )
        if len(self.style_image_files) > 0:
            self.style_image_files = self._adjust_data_length(
                self.style_image_files, self.data_length, self.data_repeat
            )
        self._differ_length_preprocess_for_data()

        # ===== 전처리 모듈 ======
        self.cropper = MTCNN(
            self.resolution, margin=self.resolution // 2, post_process=False
        )
        self.crop_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-15, 15)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.no_crop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=False,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-15, 15)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self._gender = {"f": 0, "m": 1}

    def __len__(self):
        return len(self.content_image_files)

    def __getitem__(self, idx) -> [torch.Tensor, torch.Tensor]:
        if self.same_length is False:
            if self.content_longer is True:
                content_idx = idx
                style_idx = idx % self.shorter_length
            else:
                content_idx = idx % self.shorter_length
                style_idx = idx
        else:
            content_idx = idx
            style_idx = idx

        content_image_file = self.content_image_files[content_idx]
        style_image_file = self.style_image_files[style_idx]

        content_image_file = str(content_image_file)
        style_image_file = str(style_image_file)

        content_image = self.image_process(content_image_file, do_crop=True)
        style_image = self.image_process(style_image_file, do_crop=False)
        # blur_image = self.blur_image(character_image_file)
        return [content_image, style_image]

    def image_process(
        self, image_path, do_crop: bool = True, do_transform: bool = True
    ):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if do_crop:
            image = cv2.resize(image, (512, 512))
            image = self.mtcnn_crop(image) / 255
            if do_transform:
                image = self.crop_transform(image)  # only for real image
        else:
            image = self.no_crop_transform(image)
        return image

    def mtcnn_crop(self, image):
        cropped_image = self.cropper(image)

        if cropped_image is None:  # 빈 파일일 경우.
            image = cv2.resize(image, (self.resolution, self.resolution))
            cropped_image = self.no_crop_transform(image)
        return cropped_image

    def _differ_length_preprocess_for_data(self):
        """
        길이가 다른 데이터셋을 위한 전처리입니다.
        e.g. len(dataset1) == 5
             len(dataset2) == 3

            __getitem__(self, idx):
                data1 = dataset1[idx]
                data2 = dataset2[idx % (len(dataset2) + 1)]

                when idx == 4:
                    idx % (len(dataset2) + 1) == 0
                when idx == 5:
                    idx % (len(dataset2) + 1) == 1
        """
        content_image_length = len(self.content_image_files)
        style_image_length = len(self.style_image_files)

        if content_image_length - style_image_length > 0:
            self.same_length = False
            self.content_longer = True
            self.shorter_length = style_image_length

        elif content_image_length - style_image_length < 0:
            self.same_length = False
            self.content_longer = False
            self.shorter_length = content_image_length
        else:
            self.same_length = True
            self.content_longer = False

    def _convert_resolution_type_to_int(
        self, resolution: Union[int, float, list, tuple]
    ) -> int:
        if isinstance(resolution, (list, tuple)):
            resolution = resolution[0]
        return int(resolution)

    def blur_image(self, img_path):
        # 이미지 로드 (컬러로 읽기)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽기 때문에 RGB로 변경
        img = cv2.resize(img, (self.resolution, self.resolution))

        # 1. Canny Edge Detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # 2. Edge Dilation
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # 3. Gaussian Smoothing in the dilated edge regions
        for i in range(3):  # RGB 각 채널에 대해
            img[:, :, i][dilated_edges > 0] = gaussian_filter(
                img[:, :, i][dilated_edges > 0], sigma=1
            )

        # 이미지를 텐서로 변환
        tensor_img = (
            torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        )  # HWC -> CHW, [0, 255] -> [0, 1]

        # [-1, 1] 범위로 정규화
        tensor_img = tensor_img * 2.0 - 1.0

        return tensor_img
