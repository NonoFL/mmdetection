import torch
from mmdet.models.dense_heads.fcos_head import FCOSHead

self = FCOSHead(11,7)
feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
cls_score, bbox_pred, centerness = self.forward(feats)
