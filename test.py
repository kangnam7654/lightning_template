import pandas as pd

from dataset import build_dataloader
from models.resnet import Classifier
from utils import train_utils
from utils.common.common import load_config
from utils.common.project_paths import GetPaths


def main(cfg):
    train_utils.MakeDataFrame()  # data 폴더를 읽고 csv를 만듭니다.
    train_csv = pd.read_csv(GetPaths.get_data_folder("train.csv"))  # train.csv 로딩
    train_df, valid_df = train_utils.split_csv(train_csv)  # 학습, 검증세트 분할

    # data loaders
    train_loader = build_dataloader(train_df, cfg=cfg, mode="train")
    valid_loader = build_dataloader(valid_df, cfg=cfg, mode="valid")

    # model
    model = Classifier(cfg)

    # callbacks
    logger = train_utils.CSVLogger(
        path=cfg.TRAIN.LOGGING_SAVE_PATH, sep=cfg.TRAIN.LOGGING_SEP
    )
    checkpoint = train_utils.SaveCheckPoint(path=cfg.TRAIN.MODEL_SAVE_PATH)
    early_stopping = train_utils.EarlyStopping(
        patience=cfg.TRAIN.EARLYSTOP_PATIENT, verbose=True
    )

    for epoch in range(cfg.TRAIN.EPOCHS):
        # train
        train_avg_loss, train_total_loss, train_acc = train_utils.share_loop(
            epoch, model, train_loader, mode="train"
        )
        # validation
        valid_avg_loss, valid_total_loss, valid_acc = train_utils.share_loop(
            epoch, model, valid_loader, mode="valid"
        )

        # TBD: list에 담기
        # list_train_loss.extend(train_total_loss)
        # list_valid_loss.extend(valid_total_loss)
        # list_avg_train_loss.append(train_avg_loss)
        # list_avg_valid_loss.append(valid_avg_loss)

        results = [epoch, train_avg_loss, valid_avg_loss, train_acc, valid_acc]
        train_utils.print_result(result=results)  # 결과출력
        logger.logging(results)  # 로깅
        checkpoint(valid_avg_loss, model)  # 체크포인트 저장
        early_stopping(valid_avg_loss)  # 얼리스탑
        if early_stopping.early_stop:
            break
            