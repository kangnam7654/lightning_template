import cv2
import numpy as np
import torch


def invert_image_process(tensor: torch.Tensor, convert_bgr: bool = True) -> np.uint8:
    """
    prediction 결과, Tanh 를 거쳐 (-1, 1)로 나온 결과를 다시 이미지로 되돌리는 함수입니다.
    """
    tensor = tensor.clone()

    if len(tensor.shape) == 4:
        images = []
        for t in tensor:
            tensor = t.permute(1, 2, 0)
            image = tensor.detach().cpu().numpy()
            image = np.clip(image, -1, 1)  # Assert
            image = (image + 1) * 127.5  # (-1, 1) -> (0, 255)
            image = image.astype(np.uint8)
            if convert_bgr:  # To cv2 order
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
        images = np.stack(images, axis=0)
        return images

    elif len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)
        image = tensor.detach().cpu().numpy()
        image = np.clip(image, -1, 1)  # Assert
        image = (image + 1) * 127.5  # (-1, 1) -> (0, 255)
        image = image.astype(np.uint8)
        if convert_bgr:  # To cv2 order
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    else:
        raise ValueError(f"Shape를 확인하세요. 현재 shape : {tensor.shape}")
