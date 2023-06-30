import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class BasePipeline(pl.LightningModule):
    def __init__(self, model=None, manual_ckpt_save_path=None):
        super().__init__()
        self.model = model
        self.manual_ckpt_save_path = manual_ckpt_save_path
        self.manual_best_saved = None
        # TODO
        # if self.manual_last_saved:
        #   self.manual_last_saved = self.rename_ckpt_path(for_last=True)

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

    def tensor_to_image(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.clone()
            tensor = tensor[0]
        elif len(tensor.shape) == 3:
            tensor = tensor.clone()
        else:
            raise ValueError("텐서 길이 확인")

        tensor = torch.clamp(tensor, -1, 1)
        tensor = tensor.detach().cpu()
        tensor = tensor.add(1).mul(127.5)
        image = tensor.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def save_in_model_weight(self, save_path):
        torch.save(self.model.state_dict(), save_path)
