import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDatamodule(Dataset, ABC):
    def __init__(self):
        super().__init__()
        self.image_default_extensions = ["jpg", "jpeg", "png", "bmp", "webp"]
        self.label_default_extensions = [".npy", ".json"]
        self.discrete_dim = 3

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def image_process(self, image_path):
        pass

    def glob_files(
        self,
        root_dir: Union[None, str, Path],
        extensions: Union[str, list, tuple],
    ) -> list:
        if root_dir is None:
            warnings.warn(f"root_dir 인자가 현재 {root_dir} 입니다. 빈 리스트를 반환합니다.")
            return []
        else:
            root_dir = self.assert_path(root_dir)

        # ===== 파일 확장자 확인 =====
        if isinstance(extensions, (list, tuple)):
            files = []
            for extension in extensions:
                globed = sorted(root_dir.rglob(f"*{extension}"))  # Recursive
                files.extend(globed)

        else:  # 확장자가 하나인 경우
            files = root_dir.rglob(f"*{extensions}")  # Recursive
        return files

    def label_process(
        self, label_file: Union[str, Path], type_sample: torch.Tensor = None
    ):
        label = np.load(label_file)
        label = torch.from_numpy(label)  # to Tensor
        if type_sample is not None:
            label = label.type_as(type_sample)  # Type 변경
        return label

    def _extensions_to_list(
        self, extension: Union[str, list, tuple], default_extensions: Union[list, tuple]
    ) -> Union[list, tuple]:
        """
        확장자를 다루는 메서드입니다. 입력된 확장자를 list형식으로 return합니다.
        Args
            extension : 이 class에서 사용할 확장자입니다. None일 경우, default_extension을 리턴합니다.
            default_extension: 클래스에 미리 설정된 확장자를 입력합니다.
        """
        if extension is None:
            return default_extensions
        elif isinstance(extension, str):
            return [extension]
        elif isinstance(extension, (list, tuple)):
            return extension
        else:
            raise ValueError("extension는 str, list, tuple 중 하나여야 합니다.")

    def _adjust_data_length(self, data_files, data_length, data_repeat):
        if data_length:
            data_files = data_files[:data_length] * data_repeat
        else:
            data_files = data_files * data_repeat
        return data_files

    def get_label_path(
        self, data_path: Union[str, Path], label_extensions: Union[list, tuple]
    ) -> Path:
        """
        데이터(아마 이미지) 경로에서 같은 이름을 가진 label을 불러오는 함수입니다.
        """
        data_path = self.assert_path(data_path)
        for extension in label_extensions:
            if data_path.with_suffix(extension).is_file():
                label_path = data_path.with_suffix(extension)
                return label_path
        raise ValueError(f"{data_path}에 해당하는 라벨 파일이 없습니다.")

    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size=(512, 512)):
        # 원본 이미지의 크기
        height, width, _ = image.shape
        target_w, target_h = target_size
        base_pic = np.zeros((target_h, target_w, 3), np.uint8)
        ratio_h = target_h / height
        ratio_w = target_w / width
        if ratio_w < ratio_h:
            new_size = (int(width * ratio_w), int(height * ratio_w))

        else:
            new_size = (int(width * ratio_h), int(height * ratio_h))
        new_w, new_h = new_size
        image = cv2.resize(image, dsize=new_size)
        base_pic[
            int(target_h / 2 - new_h / 2) : int(target_h / 2 + new_h / 2),
            int(target_w / 2 - new_w / 2) : int(target_w / 2 + new_w / 2),
            :,
        ] = image
        return base_pic

    def assert_path(self, path: Union[str, Path]) -> Path:
        """
        str이나 Path로 들어온 경로를 Path로 바꿔줍니다.
        Path나 str이 아닌경우 에러를 뱉습니다.
        """
        if isinstance(path, Path):
            return path
        elif isinstance(path, str):
            return Path(path)
        else:
            raise TypeError(
                f"path의 type은 str 혹은 pathlib.Path 여야합니다. type을 확인하세요. 현재 type: {type(path)}"
            )
