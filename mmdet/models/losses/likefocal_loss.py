import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@mmcv.jit(derivate=True, coderize=True)
def likefocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    assert pred.size() == target.size()
    # pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    label = torch.zeros_like(target)
    label[target>0] = 1
    label.requires_grad=True

    focal_weight = alpha * (1- target).pow(gamma) * (label>0.0).float() + \
        (1-alpha)* (label<1).float()
    
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module()
class Likefocalloss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):  
        super(Likefocalloss, self).__init__()
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * likefocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
