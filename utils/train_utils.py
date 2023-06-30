import time
from glob import glob

import os
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split

from utils.common.project_paths import GetPaths
from pathlib import Path


def print_result(result):
    """
    결과를 print하는 함수 입니다.
    :param result: list를 input으로 받아 print합니다.
    :return:
    """
    epoch, train_loss, valid_loss, train_acc, valid_acc = result
    print(
        f"[epoch{epoch}] train_loss: {round(train_loss, 3)}, valid_loss: {round(valid_loss, 3)}, train_acc: {train_acc}%, valid_acc: {valid_acc}%"
    )


def split_csv(df, test_size=0.2, seed=None):
    """
    학습 데이터셋과 검증 데이터셋으로 나누는 함수입니다.
    :param df:
    :param test_size:
    :param seed:
    :return:
    """
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["cls"]
    )
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    return train_df, valid_df


def share_loop(epoch, model, data_loader, mode="train"):
    """
    학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
    :param epoch:
    :param model:
    :param data_loader:
    :param mode: 'train', 'valid', 'test' 중 하나의 값을 받아 loop를 진행합니다.
    :return: average_loss(float64), total_losses(list), accuracy(float)
    """
    count = 0
    correct = 0
    total_losses = []

    mode = mode.lower()
    if mode == "train":
        model.train()
        for batch in tqdm.tqdm(data_loader, desc=f"{mode} {epoch}"):
            data, label = batch
            out = model(data)

            # accuracy 계산
            predicted = torch.argmax(torch.softmax(out, dim=1), dim=1)
            count += label.size(0)
            correct += (predicted == label).sum().item()

            # 역전파
            model.optimizer.zero_grad()
            loss = model.criterion(out, label)
            total_losses.append(loss.item())
            loss.backward()
            model.optimizer.step()

    elif mode in ['valid', 'test']:  # valid & test
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc=f"{mode} {epoch}"):
                data, label = batch
                out = model(data)

                # accuracy 계산
                predicted = torch.argmax(torch.softmax(out, dim=1), dim=1)
                count += label.size(0)
                correct += (predicted == label).sum().item()
                loss = model.criterion(out, label)
                total_losses.append(loss.item())
    else:
        raise Exception(f'mode는 train, valid, test 중 하나여야 합니다. 현재 mode값 -> {mode}')

    avg_loss = np.average(total_losses)
    accuracy = 100 * correct / count  # Accuracy 계산
    return avg_loss, total_losses, accuracy


class CSVLogger:
    def __init__(self, path=None, sep=','):
        self.df = self.define_df()
        self.sep = sep
        self.path = path
        self.set_path()

    @staticmethod
    def define_df():
        columns = ["epoch", "train_loss", "valid_loss", "train_acc", "valid_acc"]
        df = pd.DataFrame(columns=columns)
        return df

    def logging(self, results):
        self.df.loc[len(self.df)] = results  # 로깅
        self.df.to_csv(self.path, index=False, sep=self.sep)

    def set_path(self):
        if self.path is None:
            self.path = Path('./log.txt')
        else:
            self.path = Path(self.path)

        if not os.path.isdir(self.path.parent):
            os.makedirs(self.path.parent, exist_ok=True)

        if not len(self.path.suffixes):
            self.path.replace(self.path.with_suffix('.txt'))  # 파일 확장자가 없을 경우, txt를 기본 확장자로 지정


class SaveCheckPoint:
    """ 모델 학습 진행상황을 저장합니다. """
    def __init__(self, path=None, verbose=False, delta=0, mode='min'):
        """
        :param path: 모델을 저장할 경로입니다. None일 경우, YYMMDDHHMM.pth 로 저장이 됩니다.
        :param verbose: 모델을 저장할 경우, 해당 내용을 출력할지 정하는 변수입니다.
        :param delta: monitor할 score의 최소 improvement를 정하는 변수입니다. delta 이상의 improvement가 있어야 checkpoint가 저장이 됩니다.

        :param mode: improvement의 방향성을 결정합니다. 'min', 'max'의 값을 받으며,
                     min일 경우 score의 감소가 improvement, max일 경우 score의 증가가 improvement입니다.
        """
        self.verbose = verbose
        self.path = path
        self.delta = delta
        self.mode = mode.lower()
        self.best_score = np.Inf
        self.val_loss_min = np.Inf

    def __call__(self, score, model):
        score_ = abs(score)
        if self.mode == 'min':
            improve_condition = self.best_score - score_ > self.delta
        elif self.mode == 'max':
            improve_condition = score_ - self.best_score > self.delta
        else:
            raise Exception('mode의 변수는 min 또는 max여야 합니다.')

        if improve_condition:
            self.best_score = score_
            self.save_ckpt(model)

    def save_ckpt(self, model):
        """
        checkpoint를 저장하는 메서드입니다.
        :param model:
        :return:
        """
        t = time.strftime("%y%m%d%H%M")
        if self.path is None:
            self.path = f'./ckpt/model/{t}.pth'
        else:
            pass
        directory = GetPaths.path.split(self.path)[0]
        if not GetPaths.path.isdir(directory):
            GetPaths.makedirs(directory, exist_ok=True)
        if self.verbose:
            self.verbose_print()
        torch.save(model.state_dict(), self.path)  # state dict로 저장

    def verbose_print(self):
        print(f"best score가 {round(self.best_score, 3)}로 갱신되었습니다. 모델을 저장합니다...")


class EarlyStopping:
    """주어진 patience 이후로 score가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=5, verbose=True, delta=0, mode='min'):
        """
        Args:
            patience (int): score가 개선된 후 기다리는 기간
                            Default: 5
            verbose (bool): True일 경우 각 score의 개선 사항 메세지 출력
                            Default: True
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            mode (str): 'min', 'max'를 선택하여 score의 개선 방향을 선택
                            Default: min
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.mode = mode.lower()
        self.delta = delta
        self.best_score = np.Inf
        self.early_stop = False

    def __call__(self, score):
        score_ = abs(score)
        if self.mode == 'min':
            improve_condition = self.best_score - score_ > self.delta
        elif self.mode == 'max':
            improve_condition = score_ - self.best_score > self.delta
        else:
            raise Exception('mode의 변수는 min 또는 max여야 합니다.')

        if not improve_condition:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.verbose_print()
        else:
            self.counter = 0
            self.best_score = score_

    def verbose_print(self):
        print(f"score가 {self.counter}회 동안 개선되지 않았습니다. 학습을 중지합니다.")


class MakeDataFrame:
    def __init__(self, train_csv='train.csv', test_csv='test.csv'):
        self.train_image_list = self.load_file_list(folder='train')
        self.test_image_list = self.load_file_list(folder='test')
        self.name_class_map = self.read_classes()
        self.train_df = self.make_df(image_list=self.train_image_list, train=True)
        self.test_df = self.make_df(image_list=self.test_image_list, train=False)
        if not os.path.isfile(GetPaths.get_data_folder('train.csv')):
            self.train_df.to_csv(GetPaths.get_data_folder(train_csv), index=False)
            self.test_df.to_csv(GetPaths.get_data_folder(test_csv), index=False)
        else:
            pass

    @staticmethod
    def load_file_list(folder='train'):
        image_extension = ['png', 'jpg', 'jpeg']
        image_list = []
        for ext in image_extension:
            image_list.extend(glob(GetPaths.get_data_folder(f'{folder}', '**', f'*.{ext}'), recursive=True))
        return image_list

    @staticmethod
    def read_classes():
        class_folders = sorted(glob(GetPaths.get_data_folder('train', '*')))
        name_class_map = {}
        for idx, folder in enumerate(class_folders):
            cls_nm = folder.split(os.path.sep)[-1]
            name_class_map[cls_nm] = idx
        return name_class_map

    def make_df(self, image_list, train=True):
        maps = {}
        for idx, file in enumerate(image_list):
            splt = os.path.split(file)
            image = splt[1]
            if train:
                cls_name = splt[0].split(os.path.sep)[-1]
                maps[idx] = {'image': image,
                             'cls': self.name_class_map[cls_name],
                             'cls_name': cls_name}
            else:
                maps[idx] = {'image': image}
        df = pd.DataFrame(maps).T
        return df


if __name__ == '__main__':
    pass
