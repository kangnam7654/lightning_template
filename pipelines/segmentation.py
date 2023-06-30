import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class SegmenterPipeline(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self._loop(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loop(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True
        )
        return optimizer

    def _loop(self, batch, batch_idx):
        image, segmap_gt = batch
        feature_map = self.model(image)
        segmap = self._segmap_construction(feature_map=feature_map)
        self._show(segmap, image)
        segmap_gt = torch.squeeze(segmap_gt, 1)
        loss = F.cross_entropy(segmap, segmap_gt)
        return loss

    def _segmap_construction(self, feature_map):
        feature_map_up = F.interpolate(
            feature_map, size=(512, 512), mode="bilinear", align_corners=True
        )
        seg_map = feature_map_up
        return seg_map

    def _show(self, seg_map, image):
        # seg_map = np.random.randint(0, 5, (512, 512))  # 가정: 클래스 수가 5개
        seg_map = torch.clone(seg_map).argmax(dim=1)
        image_cloned = torch.clone(image)
        image_cloned_inversed = self._norm_inverse(image_cloned)
        image_cloned_inversed = image_cloned_inversed.detach().cpu().numpy()
        image_cloned_inversed = np.transpose(image_cloned_inversed, (0, 2, 3, 1))
        image_cloned_inversed = np.clip(image_cloned_inversed, 0, 1)
        image_cloned_inversed = image_cloned_inversed * 255
        image_cloned_inversed = image_cloned_inversed.astype(np.uint8)[0]
        image_cloned_inversed = image_cloned_inversed[:, :, ::-1]
        seg_map = seg_map.detach().cpu().numpy().astype(int)
        colors = np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [127, 127, 0],
                [127, 0, 127],
                [255, 255, 255],
            ]
        )
        seg_rgb = colors[seg_map].astype(np.uint8)[0]

        img_concat = cv2.hconcat([image_cloned_inversed, seg_rgb])
        # cv2.imwrite()
        cv2.imshow("", img_concat)  #
        cv2.waitKey(1)

    def _norm_inverse(
        self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        for t, m, s in zip(tensor, mean, std):  # For each channel
            t.mul_(s).add_(m)  # multiply by std and add mean
        return tensor
