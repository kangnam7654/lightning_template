import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class BasePipeline(pl.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def _he_init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    def _xavier_init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)


    def apply_he_init(self):
        self.model.apply(self._he_init_weights)
    
    def apply_xavier_init(self):
        self.model.apply(self._xavier_init_weights)
        
    def _compute_criterion(self, y, y_hat):
        loss = self.criterion(y, y_hat)
        return loss

    def model_tensor_to_image(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.clone()
            tensor = tensor[0]
        elif len(tensor.shape) == 3:
            tensor = tensor.clone()
        else:
            raise ValueError("텐서 길이 확인")

        tensor = tensor.detach().cpu()
        image = tensor.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        image = image * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def save_in_model_weight(self, save_path):
        torch.save(self.model.state_dict(), save_path)
