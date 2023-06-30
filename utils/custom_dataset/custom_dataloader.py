import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(ROOT_DIR.absolute())

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from utils.custom_dataset.custom_dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    """Data module of Petfinder profiles."""

    def __init__(
        self, train_df=None, valid_df=None, test_df=None, cfg=None, transform=None
    ):
        super().__init__()
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df
        self._cfg = cfg
        self.trans = transform

    def __create_dataset(self, mode="train"):
        if mode == "train":
            return CustomDataset(
                self._train_df, train=True, transform=self.trans["train"]
            )
        elif mode == "valid":
            return CustomDataset(
                self._valid_df, train=True, transform=self.trans["valid"]
            )
        elif mode in ["test", "predict"]:
            return CustomDataset(
                self._test_df, train=False, transform=self.trans["test"]
            )
        else:
            raise Exception("mode error")

    def train_dataloader(self):
        dataset = self.__create_dataset("train")
        return DataLoader(dataset)

    def val_dataloader(self):
        dataset = self.__create_dataset("valid")
        return DataLoader(dataset)

    def predict_dataloader(self):
        dataset = self.__create_dataset("predict")
        return DataLoader(dataset)

    def test_dataloader(self):
        dataset = self.__create_dataset("test")
        return DataLoader(dataset)
