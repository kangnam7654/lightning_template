import numpy as np
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
from torchvision.transforms import v2

from kangnam_packages.dataset_modules.base import BaseDatamodule


class StyleGan2Datamodule(BaseDatamodule):
    def __init__(
        self,
        data_dir: str,
        resolution=(512, 512),
        scale_range=(-1, 1),
        return_label=True,
        transform=None,
        image_extension=None,
        label_extension=None,
        data_length=None,
        data_repeat=1,
    ):
        """
        Args:
            data_dirs (str or list, tuple): Image directory.
            transform (callable, optional): To apply Transforms.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_extensions = self._extensions_to_list(
            image_extension, self.image_default_extensions
        )
        self.label_extensions = self._extensions_to_list(
            label_extension, self.label_default_extensions
        )
        self.data_files = self.glob_files(
            self.data_dir, extensions=self.image_extensions
        )
        self.data_files = self._adjust_data_length(
            data_files=self.data_files,
            data_truncation=data_length,
            data_repeat=data_repeat,
        )

        self.scale_range = scale_range
        self.resolution = resolution
        self._resolution_convert()
        self.return_label = return_label
        if transform is None:
            self.transform = v2.Compose(
                [
                    v2.ToTensor(),
                    v2.RandomRotation((-20, 20), fill=[0, 0, 0]),
                    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    v2.RandomHorizontalFlip(),
                    v2.RandomApply(
                        [
                            v2.CenterCrop(
                                int(
                                    np.random.randint(
                                        self.resolution[0] // 2, self.resolution[0]
                                    )
                                )
                            ),
                            v2.Resize(self.resolution[0]),
                        ]
                    ),
                ]
            )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image_path = self.data_files[idx]
        image = self.image_process(image_path)

        if not self.return_label:
            return image
        else:
            label_path = self.get_label_path(
                data_path=image_path, label_extensions=self.label_extensions
            )
            label = self.label_process(label_path)
            label = label.float()
        return [image, label]

    def image_process(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_with_padding(image, target_size=self.resolution)
        image = self.transform(image)
        return image

    def _resolution_convert(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            pass  # DO NOTHING

        else:
            raise ValueError(
                f"Check resoution! Current arg of resoution : {self.resolution}"
            )
