import cv2

from utils.common.project_paths import GetPaths

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    if mode in ['train', 'valid']:
        param = True
    else:
        param = False
    trans = get_transforms(cfg)[mode]
    dataset = CustomDataset(df=df, transform=trans, train=param)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=param, drop_last=False)
    return dataloader


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
                A.LongestMaxSize(max_size=cfg.DATA.IMAGE_SIZE),
                A.PadIfNeeded(min_height=cfg.DATA.IMAGE_SIZE, min_width=cfg.DATA.IMAGE_SIZE,
                              border_mode=cv2.BORDER_CONSTANT, p=0.5),
                # A.RandomCrop(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, p=0.5),
                A.Resize(cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE),
                ToTensorV2()
            ]),
        'valid':
            A.Compose([
                A.Normalize(),
                A.LongestMaxSize(max_size=cfg.DATA.IMAGE_SIZE),
                A.PadIfNeeded(min_height=cfg.DATA.IMAGE_SIZE, min_width=cfg.DATA.IMAGE_SIZE,
                              border_mode=cv2.BORDER_CONSTANT, p=1),
                ToTensorV2()
            ]),
        'test':
            A.Compose([
                A.Normalize(),
                A.LongestMaxSize(max_size=cfg.DATA.IMAGE_SIZE),
                A.PadIfNeeded(min_height=cfg.DATA.IMAGE_SIZE, min_width=cfg.DATA.IMAGE_SIZE,
                              border_mode=cv2.BORDER_CONSTANT, p=1),
                ToTensorV2()
            ])
    }
    return transforms


class CustomDataset(Dataset):
    """
    Dataset을 만드는 클래스입니다.
    param train: train, validation에는 True를, Inference에는 False를 주면 됩니다.
    """
    def __init__(self, df, transform, train=True):
        self.raw_data = df
        self.train = train
        self.df = self._preprocess()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'image']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']

        if self.train:
            label = self.df.loc[idx, 'cls']
            return image, label
        else:
            return image

    def _preprocess(self):
        """
        데이터프레임 전처리 함수입니다.
        :return:
        """
        df = self.raw_data.copy()
        image_with_path = []

        if self.train:
            for file_name, cls_name in zip(self.raw_data['image'], self.raw_data['cls_name']):
                image_with_path.append(GetPaths.get_data_folder('train', f'{cls_name}', file_name))
        else:
            for file_name in self.raw_data['image']:
                image_with_path.append(GetPaths.get_data_folder('test', file_name))
        df['image'] = image_with_path
        return df


class PetFinderDataModule(LightningDataModule):
    """Data module of Petfinder profiles."""
    def __init__(self, train_df=None, valid_df=None, test_df=None, cfg=None):
        super().__init__()
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df
        self._cfg = cfg
        self.trans = get_default_transforms()
        self.ttas = get_tta_transforms()

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return PetFinderDataset(self._train_df, train=True, transform=self.trans['train'])
        elif mode == 'valid':
            return PetFinderDataset(self._valid_df, train=True, transform=self.trans['valid'])
        elif mode == 'test':
            return PetFinderDataset(self._test_df, train=False, transform=self.trans['test'], tta=self.ttas)
        elif mode == 'predict':
            return PetFinderDataset(self._train_df, train=False, predict=True, transform=self.trans['test'], tta=self.ttas)

    def train_dataloader(self):
        dataset = self.__create_dataset('train')
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset('valid')
        return DataLoader(dataset, **self._cfg.valid_loader)

    def predict_dataloader(self):
        dataset = self.__create_dataset('predict')
        return DataLoader(dataset, **self._cfg.test_loader)

    def test_dataloader(self):
        dataset = self.__create_dataset('test')
        return DataLoader(dataset, **self._cfg.test_loader)





if __name__ == '__main__':
    pass
