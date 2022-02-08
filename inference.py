import pandas as pd
import torch
import tqdm

from dataset import build_dataloader
from models.resnet import Classifier
from utils.common.common import load_config
from utils import train_utils
from utils.common.project_paths import GetPaths


def main(cfg):
    df_make = train_utils.MakeDataFrame()  # DataFrame을 만듭니다.
    cls_map = df_make.name_class_map  # class_name을 key로, class를 value로 가지는 dictionary 입니다.

    test_csv = pd.read_csv(GetPaths().get_data_folder('test.csv'))
    test_loader = build_dataloader(test_csv, cfg=cfg, mode='test')

    model = Classifier(cfg)  # 모델 불러오기

    all_preds = []
    all_probs = []
    for image in tqdm.tqdm(test_loader):
        out = model(image)
        softmaxed = torch.softmax(out, dim=1)
        preds = torch.argmax(softmaxed, dim=1)  # 예측값
        probs = torch.max(softmaxed, dim=1).values  # 확률값
        all_preds.append(preds)
        all_probs.append(probs)
    all_preds_ = torch.cat(all_preds, dim=0)
    all_probs_ = torch.cat(all_probs, dim=0)
    # 결과 저장
    result_csv = test_csv.copy()
    result_csv['cls'] = all_preds_
    result_csv['cls_name'] = result_csv['cls'].replace(cls_map.values(), cls_map.keys())
    result_csv['prob'] = all_probs_.detach().numpy()
    result_csv.to_csv('sample_result.csv')


if __name__ == '__main__':
    config = load_config(GetPaths.get_configs_folder('resnet18.yaml'))
    main(config)
