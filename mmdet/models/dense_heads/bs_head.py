import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply
from mmdet.core.bbox import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12




@HEADS.register_module()
class BSHead(FCOSHead):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, )