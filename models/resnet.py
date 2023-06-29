import torch.nn as nn
import torchvision.models as models


class ResNet152FeatureExtractor(nn.Module):
    def __init__(self, weights=True, do_eval=True):
        super(ResNet152FeatureExtractor, self).__init__()
        self.feature_extractor = self.build_extractor(weights, do_eval)
        # self.pool = nn.AdaptiveAvgPool2d(2)

    def build_extractor(self, weights, do_eval):
        resnet = models.resnet152(weights=weights)  # 사전 훈련된 ResNet152 모델 불러오기
        feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # 마지막 선형 계층을 제외한 모든 계층을 가져옵니다.
        if do_eval is True:
            feature_extractor = feature_extractor.eval()
        else:
            pass  # do nothing
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        # features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, weights=True, do_eval=True):
        super().__init__()
        self.feature_extractor = self.build_extractor(weights, do_eval)
        # self.pool = nn.AdaptiveAvgPool2d(2)

    def build_extractor(self, weights, do_eval):
        resnet = models.resnet50(weights=weights)  # 사전 훈련된 ResNet152 모델 불러오기
        feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # 마지막 선형 계층을 제외한 모든 계층을 가져옵니다.
        if do_eval is True:
            feature_extractor = feature_extractor.eval()
        else:
            pass  # do nothing
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        # features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features

class Swinv2Encoder(nn.Module):
    def __init__(self, do_eval=True):
        super().__init__()
        self.feature_extractor = self.build_extractor(do_eval)
        # self.pool = nn.AdaptiveAvgPool2d(2)

    def build_extractor(self, do_eval):
        resnet = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)  # 사전 훈련된 ResNet152 모델 불러오기
        feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # 마지막 선형 계층을 제외한 모든 계층을 가져옵니다.
        if do_eval is True:
            feature_extractor = feature_extractor.eval()
        else:
            pass  # do nothing
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        # features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features