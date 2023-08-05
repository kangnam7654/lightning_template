import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import cv2


def main():
    for _ in range(1000000):
        x = torch.rand(4, 3, 224, 224)
        grid = make_grid(x, normalize=False).permute(1, 2, 0)
        grid = grid.numpy()
        # grid = grid
        # grid = grid.astype(np.uint8)
        cv2.imshow("", grid)
        cv2.waitKey(1)

    pass


if __name__ == "__main__":
    main()
