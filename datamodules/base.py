from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data.preprocess.image_process import image_preprocess


class BaseDatamodule(Dataset, ABC):
    image_default_suffixes = ["jpg", "jpeg", "png", "bmp", "webp"]
    label_default_suffixes = [".npy"]
    discrete_dim = 3

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def image_process(self):
        pass

    def glob_files(self, root_dir: Union[Path, str], suffixes: Union[str, list, tuple]):
        files = []
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if (
            not isinstance(suffixes, str) and len(suffixes) > 1
        ):  # suffixes가 list 거나 tuple 이고 크기가 1이상인 경우
            for suffix in suffixes:
                for file in root_dir.glob(f"**/*{suffix}"):
                    files.append(file)
        else:
            files = [str(file.resolve()) for file in root_dir.glob(f"**/*{suffixes}")]
        return files

    def label_process(self, label_file, type_sample=None):
        label = np.load(label_file)
        label = torch.from_numpy(label)
        if type_sample is not None:
            label = label.type_as(type_sample)
        return label

    def _parse_suffixes(
        self, suffix: Union[str, list, tuple], default_suffixes: Union[list, tuple]
    ):
        if suffix is None:
            return default_suffixes
        elif isinstance(suffix, str):
            return [suffix]
        elif isinstance(suffix, (list, tuple)):
            return suffix
        else:
            raise ValueError("suffix는 str, list, tuple 중 하나여야 합니다.")

    def _adjust_data_length(self, data_files, data_length, data_repeat):
        if data_length:
            data_files = data_files[:data_length] * data_repeat
        else:
            data_files = data_files * data_repeat
        return data_files

    def get_label_path(self, data_path, label_suffixes):
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        _cnt = 0
        for suffix in label_suffixes:
            if data_path.with_suffix(suffix).is_file():
                _cnt += 1
                label_path = data_path.with_suffix(suffix)
                break
        if _cnt == 0:
            raise ValueError(f"{data_path}에 해당하는 라벨 파일이 없습니다.")
        return label_path
