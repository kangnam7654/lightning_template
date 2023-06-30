import random
import sys
from pathlib import Path
from typing import Union

import cv2
import torch
from torchvision.transforms.functional import to_tensor

# R = Path(__file__).parents[1]
# sys.path.append(str(R.resolve()))

# from utils.data.preprocess.image_process import image_preprocess
from .base import BaseDatamodule

# from lib.facenet_torch.mtcnn import MTCNN


class CustomImageDataset(BaseDatamodule):
    def __init__(
        self,
        data_dir: str,
        resolution=(512, 512),
        scale_range=(-1, 1),
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
        # ===== 데이터 1 =====
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
        # ===== 데이터 1 끝 =====

        self.scale_range = scale_range
        self.resolution = resolution
        self._resolution_convert()
        self.z_norm = z_norm
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image_path = self.data_files[idx]
        image = self.image_process(str(image_path))
        image = self._low_frequency_process(image)

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

    def image_process(
        self,
        image_path,
    ):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.resolution)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = to_tensor(image)
        image = image.sub(0.5).div(0.5)
        return image

    def _low_frequency_process(self, image):
        if self.transform is not None:
            image = self.transform(image)
        if self.z_norm:
            image = self.z_score_norm(image)
        return image

    def z_score_norm(self, tensor):
        mean = torch.mean(tensor)
        std = torch.max(torch.std(tensor), torch.tensor([1e-7]))
        normalized = (tensor - mean) / std
        return normalized

    def _resolution_convert(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            pass  # DO NOTHING

        else:
            raise ValueError(f"resolution을 확인하세요. 현재값 : {self.resolution}")


class MainDataset(BaseDatamodule):
    def __init__(
        self,
        real_image_dir: str,
        character_image_dir: str,
        resolution=160,
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
        self.real_image_dir = real_image_dir
        self.character_image_dir = character_image_dir
        self.resolution = self._convert_resolution_input_type(resolution)
        self.data_length = data_length
        self.data_repeat = data_repeat
        self.image_suffixes = self._parse_suffixes(
            image_suffix, self.image_default_suffixes
        )
        self.label_suffixes = self._parse_suffixes(
            label_suffix, self.label_default_suffixes
        )
        self.real_image_files = self.glob_files(real_image_dir, self.image_suffixes)
        self.character_image_files = self.glob_files(
            character_image_dir, self.image_suffixes
        )
        self.real_image_files = self._adjust_data_length(
            self.real_image_files, self.data_length, self.data_repeat
        )
        self.character_image_files = self._adjust_data_length(
            self.character_image_files, self.data_length, self.data_repeat
        )
        self._augment_less_data()
        self.cropper = MTCNN(self.resolution, margin=int(0.3 * self.resolution))

    def __len__(self):
        return len(self.real_image_files)

    def __getitem__(self, idx):
        real_image_file = self.real_image_files[idx]
        character_image_file = self.character_image_files[idx]
        label_file = self.get_label_path(character_image_file, self.label_suffixes)

        real_image_file = str(real_image_file)
        character_image_file = str(character_image_file)
        label_file = str(label_file)

        real_image = self.image_process(real_image_file)
        character_image = self.image_process(character_image_file, is_real_image=False)
        label = self.label_process(label_file, type_sample=character_image)
        return real_image, character_image, label

    def image_process(self, image_path, is_real_image=True):
        return self.mtcnn_crop(image_path, is_real_image=is_real_image)

    def mtcnn_crop(self, image_path, is_real_image=True):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_real_image:
            cropped_image = self.cropper(image)
            if cropped_image is None:
                image = cv2.resize(image, (self.resolution, self.resolution))
                image = torch.from_numpy(image)
                image = (image / 127.5) - 1
                cropped_image = torch.permute(image, (2, 0, 1))
        else:
            image = cv2.resize(image, (self.resolution, self.resolution))
            image = torch.from_numpy(image)
            image = (image / 127.5) - 1
            cropped_image = torch.permute(image, (2, 0, 1))
        return cropped_image

    def recursive_copy(self, data, target_length):
        if len(data) >= target_length:
            return data
        else:
            return self.recursive_copy(data + data, target_length)

    def _augment_less_data(self):
        l1 = len(self.real_image_files)
        l2 = len(self.character_image_files)
        sub = abs(l1 - l2)

        if sub != 0:
            if l1 >= l2:  # files2 가 더 적음
                tmp_data = self.recursive_copy(self.character_image_files.copy(), sub)
                sampled = random.sample(tmp_data, sub)
                self.character_image_files.extend(sampled)
            else:
                tmp_data = self.recursive_copy(self.real_image_files.copy(), sub)
                sampled = random.sample(tmp_data, sub)
                self.real_image_files.extend(sampled)

    def _convert_resolution_input_type(
        self, resolution: Union[int, float, list, tuple]
    ):
        if isinstance(resolution, (list, tuple)):
            resolution = resolution[0]
        return int(resolution)
