from numpy import square
from numpy.core.fromnumeric import reshape, squeeze
from mmdet.models.dense_heads.autoassign_head import AutoAssignHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply
from mmdet.core.bbox import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.dense_heads.fcos_head import INF, FCOSHead
from mmdet.models.dense_heads.paa_head import levels_to_images

EPS = 1e-12


class CenterPrior(nn.Module):
    def __init__(self,
                 force_topk=False,
                 topk=9,
                 num_classes=80,
                 strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list, gt_bboxes, labels,
                inside_gt_bbox_mask):
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        points_per_img = torch.cat(anchor_points_list, dim=0)

        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points,
                                       num_gts), inside_gt_bbox_mask

        exp_points_per_img = points_per_img[:,None,:].expand(points_per_img.size(0), len(gt_bboxes), 2)

        gt_center_x = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2)
        gt_center_y = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2)
        gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)     
        gt_center = gt_center[None]

        instance_center = self.mean[labels][None]
        instance_sigma = self.sigma[labels][None]
        distance = (torch.abs(exp_points_per_img - gt_center) - instance_center)**2
        center_prior = torch.exp(-distance / (2 * instance_sigma**2)).prod(dim=-1)
        center_prior[~inside_gt_bbox_mask] = 0
        return center_prior, inside_gt_bbox_mask

@HEADS.register_module()
class AutoAssignPlusHead(FCOSHead):

    def __init__(self,
                 *args,
                 force_topk=False,
                 topk=9,
                 pos_loss_weight=0.25,
                 neg_loss_weight=0.75,
                 center_loss_weight=0.75,
                 **kwargs):
        super().__init__(*args, conv_bias=True, **kwargs)
        self.center_prior = CenterPrior(
            force_topk=force_topk,
            topk=topk,
            num_classes=self.num_classes,
            strides=self.strides)
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight

    def init_weights(self):
        """Initialize weights of the head.

        In particular, we have special initialization for classified conv's and
        regression conv's bias
        """

        super(AutoAssignPlusHead, self).init_weights()
        bias_cls = bias_init_with_prob(0.02)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=4.0)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Almost the same as the implementation in fcos, we remove half stride
        offset to align with the original implementation."""

        y, x = super(FCOSHead,
                     self)._get_points_single(featmap_size, stride, dtype,
                                              device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1)
        return points

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        centerness = self.conv_centerness(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)
        bbox_pred *= stride
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        bs = len(cls_scores)
        all_num_gt = sum([len(item) for item in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        inside_gt_bbox_mask_list, bbox_targets_list = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)
        
        # inside_gt_bbox_mask = torch.cat(inside_gt_bbox_mask_list)
        center_prior_weight_list = []
        temp_inside_gt_mask_list = []
        for gt_bbox, gt_label, inside_gt_bbox_mask in zip(gt_bboxes, gt_labels, inside_gt_bbox_mask_list):
            center_prior_weight, inside_gt_bbox_mask = self.center_prior(all_level_points, gt_bbox, gt_label, inside_gt_bbox_mask)
            center_prior_weight_list.append(center_prior_weight)
        
        mlvl_points = torch.cat(all_level_points, dim=0)
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        ious_list = []
        num_points = len(mlvl_points)
        for bbox_pred, gt_bboxe, inside_gt_bbox_mask in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = distance2bbox(expand_mlvl_points,
                                               expand_bbox_pred)
            decoded_target_preds = distance2bbox(expand_mlvl_points, gt_bboxe)
            with torch.no_grad():
                ious = bbox_overlaps(
                    decoded_bbox_preds, decoded_target_preds, is_aligned=True)
                ious = ious.reshape(num_points, temp_num_gt)
                if temp_num_gt:
                    ious = ious.max(
                        dim=-1, keepdim=True).values.repeat(1, temp_num_gt)
                else:
                    ious = ious.new_zeros(num_points, temp_num_gt)
                ious[~inside_gt_bbox_mask] = 0
                ious_list.append(ious)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        
        pos_loss_list, = multi_apply(self.get_pos_loss_single, cls_scores, objectnesses, reg_loss_list, gt_labels, center_prior_weight_list)
        pos_avg_factor = reduce_mean(bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor

        neg_loss_list, = multi_apply(self.get_neg_loss_single, cls_scores, objectnesses, gt_labels, ious_list, inside_gt_bbox_mask_list)
        neg_avg_factor = sum(item.data.sum()
                             for item in center_prior_weight_list)
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor

        center_loss = []
        for i in range(len(img_metas)):

            if inside_gt_bbox_mask_list[i].any():
                center_loss.append(
                    len(gt_bboxes[i]) /
                    center_prior_weight_list[i].sum().clamp_(min=EPS))
            # when width or height of gt_bbox is smaller than stride of p3
            else:
                center_loss.append(center_prior_weight_list[i].sum() * 0)

        center_loss = torch.stack(center_loss).mean() * self.center_loss_weight
        # avoid dead lock in DDP
        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum(
            ) * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses[0].sum() * 0 + dummy_center_prior_loss

        loss = dict(
            loss_pos=pos_loss, loss_neg=neg_loss, loss_center=center_loss)

        return loss

    def get_pos_loss_single(self, cls_score, objectness, reg_loss, gt_labels, center_prior_weights):
        # p_loc = torch.exp(-reg_loss)
        # p_cls = (cls_score * objectness)[:,gt_labels]
        # p_pos = p_cls 

        # confidence_weight = (p_pos) / (p_pos).sum(0, keepdim=True).clamp(min=EPS)
        # reweight_p_pos = confidence_weight * center_prior_weights
        
        # pos_loss = F.binary_cross_entropy(reweight_p_pos, p_loc, reduction='none')
        # pos_loss = pos_loss.sum() * self.pos_loss_weight
        # return pos_loss
        # p_loc: localization confidence
        p_loc = torch.exp(-reg_loss)
        # p_cls: classification confidence
        p_cls = (cls_score * objectness)[:, gt_labels]
        # p_pos: joint confidence indicator
        p_pos = p_cls * p_loc

        # 3 is a hyper-parameter to control the contributions of high and
        # low confidence locations towards positive losses.
        confidence_weight = torch.exp(p_pos * 3)
        p_pos_weight = (confidence_weight * center_prior_weights) / (
            (confidence_weight * center_prior_weights).sum(
                0, keepdim=True)).clamp(min=EPS)
        reweighted_p_pos = (p_pos * p_pos_weight).sum(0)
        pos_loss = F.binary_cross_entropy(
            reweighted_p_pos,
            torch.ones_like(reweighted_p_pos),
            reduction='none')
        pos_loss = pos_loss.sum() * self.pos_loss_weight
        return pos_loss,

    def get_neg_loss_single(self, cls_score, objectness, gt_labels, ious, inside_gt_bbox_mask):
        num_gts = len(gt_labels)
        joint_conf = (cls_score * objectness)
        p_neg_weight = torch.ones_like(joint_conf)
        if num_gts>0:
            inside_gt_bbox_mask = inside_gt_bbox_mask.permute(1, 0)
            ious = ious.permute(1, 0)     
            foreground_idxs = torch.nonzero(inside_gt_bbox_mask, as_tuple=True)
            temp_weight = (1 / (1 - ious[foreground_idxs]).clamp_(EPS))

            def normalize(x):
                return (x - x.min() + EPS) / (x.max() - x.min() + EPS)

            for instance_idx in range(num_gts):
                idxs = foreground_idxs[0] == instance_idx
                if idxs.any():
                    temp_weight[idxs] = normalize(temp_weight[idxs])

            p_neg_weight[foreground_idxs[1],
                         gt_labels[foreground_idxs[0]]] = 1 - temp_weight

        logits = (joint_conf * p_neg_weight)
        neg_loss = (
            logits**2 * F.binary_cross_entropy(
                logits, torch.zeros_like(logits), reduction='none'))
        neg_loss = neg_loss.sum() * self.neg_loss_weight
        return neg_loss,           







    def get_targets(self, points, gt_bboxes_list, gt_labels):

        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        inside_gt_bbox_mask_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, points=concat_points)
        return inside_gt_bbox_mask_list, bbox_targets_list

    def _get_target_single(self, gt_bboxes, points):
        """Compute regression targets and each point inside or outside gt_bbox
        for a single image.

        Args:
            gt_bboxes (Tensor): gt_bbox of single image, has shape
                (num_gt, 4).
            points (Tensor): Points of all fpn level, has shape
                (num_points, 2).

        Returns:
            tuple[Tensor]: Containing the following Tensors:

                - inside_gt_bbox_mask (Tensor): Bool tensor with shape
                  (num_points, num_gt), each value is used to mark
                  whether this point falls within a certain gt.
                - bbox_targets (Tensor): BBox targets of each points with
                  each gt_bboxes, has shape (num_points, num_gt, 4).
        """
        num_points = points.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_gts:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.new_zeros((num_points, num_gts),
                                                         dtype=torch.bool)

        return inside_gt_bbox_mask, bbox_targets