import sklearn.svm
import yaml

from utils.lightning.petfinder_dataset_v1 import PetFinderDataModule
from model.swin_v1 import ClassSwin
from utils.lightning.train_modules import LightningTrainModule
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
import pandas as pd
from utils.common.project_paths import GetPaths

import torch.multiprocessing
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold


import warnings
import gc
from box import Box
import glob

######################

from utils.common.common import load_config
from utils.common.project_paths import GetPaths


def setting():
    seed_everything() # TODO: seed set
    warnings.simplefilter('ignore')
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed = config.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_weights_list():
    weights = sorted(glob.glob(GetPaths().get_project_root('result', 'swin_cutmix', '*.pth')))
    return weights


def load_csv():
    train_csv = pd.read_csv('./data/train.csv') # TODO: path 수정
    test_csv = pd.read_csv('./data/test.csv') # TODO : path 수정
    return train_csv, test_csv

def main(cfg, cutmix=False):
    setting()
    weights = load_weights_list() # TODO: Model 내부로 옮기기

    # Run the Kfolds training loop
    skf = StratifiedKFold(
        n_splits=config.train.n_folds, shuffle=True, random_state=config.train.fold_random_state
    )


    # Dataset & DataLoader
    train_csv, test_csv = load_csv()
    train_loader, valid_loader, test_loader = build_loaders() # TODO: build_loaders 수정

    # logger & callbacks
    lightning_loggers = pl_loggers.CSVLogger()
    lr_monitor = callbacks.LearningRateMonitor()
    early_stopping = callbacks.EarlyStopping()
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath=f'./result/{config.train.log_name}/',
        filename='best_loss',
        monitor='valid_rmse',
        save_top_k=1,
        mode='min',
        save_last=False,
        verbose=True
    )
    # Build Model
    model = BuildModel(cfg) # TODO: model load
    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        gpus=config.train.n_gpus,
        strategy="ddp_spawn",
        callbacks=[lr_monitor, loss_checkpoint, early_stopping],
        logger=lightning_loggers,
        precision=config.train.precision,
    )

    trainer.fit(model, train, valid)

    # 메모리 청소
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    config = load_config('configs/template.yaml')
    main(config, cutmix=True)
