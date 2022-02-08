import torch
import torch.nn as nn
import timm


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet18d', pretrained=True, num_classes=0)

    def forward(self, x):
        out = self.model(x)
        return out


class Classifier(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg  # config
        self._build_model()
        self.criterion = nn.CrossEntropyLoss().to(self.cfg.TRAIN.DEVICE)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        if self.cfg.MODEL.CHECKPOINT is not None:
            self.ckpt = torch.load(self.cfg.MODEL.CHECKPOINT)
            self.load_state_dict(self.ckpt)
        else:
            self.ckpt = None

    def forward(self, x):
        embedding = self.backbone(x)
        out = self.fc(embedding)
        return out

    def _build_model(self):
        self.backbone = ResNet18().to(self.cfg.TRAIN.DEVICE)
        n_features = list(self.backbone.parameters())[-1].size(0)
        self.fc = nn.Linear(n_features, self.cfg.MODEL.N_CLASSES).to(self.cfg.TRAIN.DEVICE)


if __name__ == '__main__':
    backbone = ResNet18()
    model = Classifier()