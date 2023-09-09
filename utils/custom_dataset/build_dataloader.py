import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(ROOT_DIR.absolute())

from data_modules.load_transform import get_transforms
from utils.custom_dataset.custom_dataloader import CustomDataModule

def build_dataloader(df, cfg, mode):
    """
    Augmentation을 불러오고 Dataset을 불러와 적용하여 Dataloader로 wrapping하는 함수입니다.

    :args
        df: 데이터 프레임
        cfg: config. 본 함수에서는 batch size를 받습니다.
        mode: 'train, valid, test'의 값들 중 하나를 받습니다.
    :return dataloader
    """
    mode = mode.lower()
    assert mode in ['train', 'valid', 'test'], 'mode의 입력값은 train, valid, test 중 하나여야 합니다.'
    # if mode in ['train', 'valid']: # ??
    #     param = True
    # else:
    #     param = False

    transform = get_transforms(cfg)[mode]
    data_module = CustomDataModule(train_df=df, valid_df=df, test_df=df, transform=transform)
    if mode == 'train':
        dataloader = data_module.train_dataloader()
    elif mode == 'valid':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    return dataloader
