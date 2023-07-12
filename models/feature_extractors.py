import torch.nn as nn
import torchvision.models as models


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, do_eval: bool = True):
        super().__init__()
        self.feature_extractor = self.build_extractor()
        if do_eval:
            self.feature_extractor = self.feature_extractor.eval()

    def build_extractor(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features
    
class ResNet50_(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = self.build_extractor()
        self.fc = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(inplace=True), nn.Linear(1024, 1, bias=False))

    def build_extractor(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class ResNet152FeatureExtractor(nn.Module):
    def __init__(self, do_eval: bool = True):
        super().__init__()
        self.feature_extractor = self.build_extractor()
        if do_eval:
            self.feature_extractor = self.feature_extractor.eval()

    def build_extractor(self):
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT
        )  # 사전 훈련된 ResNet152 모델 불러오기
        feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # 마지막 선형 계층을 제외한 모든 계층을 가져옵니다.
        return feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features


class Swinv2Encoder(nn.Module):
    def __init__(self, do_eval=True):
        super().__init__()
        self.feature_extractor = self.build_extractor(do_eval)
        # self.pool = nn.AdaptiveAvgPool2d(2)

    def build_extractor(self, do_eval):
        resnet = models.swin_v2_t(
            weights=models.Swin_V2_T_Weights.DEFAULT
        )  # 사전 훈련된 ResNet152 모델 불러오기
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

    def test():
        model = ResNet50FeatureExtractor()
        print(model)

    if __name__ == "__main__":
        test()
