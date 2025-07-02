import numpy as np
import torch
from torchvision.utils import make_grid


def concat_tensor_images(
    *tensor_images: torch.Tensor, is_rgb: bool = True
) -> np.ndarray:
    """
    Concatenate images.

    Args:
        tensor_images: Input reconstructed images. They will be concatenated in the order provided.
    Returns:
        np.ndarray: Concatenated grid of images.
    """
    images = list(tensor_images)  # shape == (n, batch, ch, h, w)
    nrow = len(images)

    # Image calibration
    aligned = []
    batch_size = tensor_images[0].shape[0]

    for idx in range(batch_size):
        for image in images:  # image == (batch, ch, h, w)
            if is_rgb:
                aligned.append(image[idx][[2, 1, 0], :, :])  # RGB to BGR
            else:
                aligned.append(image[idx])

    grid = (
        make_grid(
            aligned,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return grid
