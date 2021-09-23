import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class BSAssigner(BaseAssigner):

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               cls_scores,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        overlaps = self.iou_calculator(bboxes, gt_bboxes) 


























        # gt_bboxes = gt_bboxes[None].expand(num_bboxes, num_gts, 4)
        
        # # inside_gt_bbox_mask
        # xs = ((bboxes[:,0] + bboxes[:, 2]) / 2)
        # ys = ((bboxes[:,1] + bboxes[:, 2]) / 2)

        # xs = xs[:, None]
        # ys = ys[:, None]
        # left = xs - gt_bboxes[..., 0]
        # right = gt_bboxes[...,2] - xs
        # top = ys - gt_bboxes[..., 1]
        # bottom = gt_bboxes[..., 3] - ys
        # temp_bboxes = torch.stack((left, top, right, bottom), -1)
        # inside_gt_bbox_mask = temp_bboxes.min(-1)[0] > 0 

        # # 求IoU，不在gt框内的不算，删除
        # overlaps[~inside_gt_bbox_mask] = 0
        # num_iou_gt0 = len(torch.nonzero(overlaps>0))





