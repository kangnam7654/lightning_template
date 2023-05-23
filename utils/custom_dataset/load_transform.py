import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms(cfg):
    """
    Augmentation에 적용할 요소들을 Dictionary 형태로 감싸 Return합니다.
    :return:
    """
    transforms = {
        'train':
            A.Compose([
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.ColorJitter(p=0.8),
                A.Normalize(),
                A.LongestMaxSize(max_size=512),
                A.PadIfNeeded(min_height=512, min_width=512,
                              border_mode=cv2.BORDER_CONSTANT, p=0.5),
                # A.RandomCrop(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, p=0.5),
                A.Resize(512, 512),
                ToTensorV2()
            ]),
        'valid':
            A.Compose([
                A.Normalize(),
                A.LongestMaxSize(max_size=512),
                A.PadIfNeeded(min_height=512, min_width=512,
                              border_mode=cv2.BORDER_CONSTANT, p=1),
                ToTensorV2()
            ]),
        'test':
            A.Compose([
                A.Normalize(),
                A.LongestMaxSize(max_size=512),
                A.PadIfNeeded(min_height=512, min_width=512,
                              border_mode=cv2.BORDER_CONSTANT, p=1),
                ToTensorV2()
            ])
    }
    return transforms
