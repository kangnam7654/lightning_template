import cv2
import numpy as np
import torch
from torchvision import datasets
from pathlib import Path


def download_mnist(path):
    dataset = datasets.MNIST(path, download=True, train=True)


def read_mnist(root_dir, download=False):
    root_dir = Path(root_dir)
    if download:
        download_mnist(root_dir.resolve())
    data_path = root_dir.joinpath("MNIST", "raw")
    with open(
        data_path.joinpath("train-images-idx3-ubyte").resolve(), "rb"
    ) as f_images, open(
        data_path.joinpath("train-labels-idx1-ubyte").resolve(), "rb"
    ) as f_labels:
        # 이미지 파일 헤더 스킵
        f_images.read(16)
        # 레이블 파일 헤더 스킵
        f_labels.read(8)

        # 이미지 파일 읽어오기
        images = np.frombuffer(f_images.read(), dtype=np.uint8).reshape(-1, 28, 28)
        # 레이블 파일 읽어오기
        labels = np.frombuffer(f_labels.read(), dtype=np.uint8)
    return [images, labels]


if __name__ == "__main__":
    path = Path(__file__).parents[1].joinpath("data")
    read_mnist(path.resolve())
