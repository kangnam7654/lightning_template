import cv2
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Dataset을 만드는 클래스입니다.
    args:

        train:
            학습시에는 True, Inference에는 False를 주면 됩니다.
        
        transform:
            albumentations나 torchvision.transform의 이미지 변환을 입력합니다.
    """
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self._preprocess()
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'image']
        image = cv2.imread(image_path)[:, :, ::-1] # BGR -> RGB

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.train:
            label = self.df.loc[idx, 'label']
            return image, label
        else:
            return image

    def _preprocess(self):
        pass
