import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from ..builder import build_bbox_coder

@BBOX_ASSIGNERS.register_module()
class NewAssigner(BaseAssigner):
    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.ignore_iof_thr = ignore_iof_thr
    def assign(self, bboxes, num_level_bboxes, 
                gt_bboxes, cls_scores, bbox_preds, 
                gt_labels=None,gt_bboxes_ignore=None):
        INF = 100000000
        bboxes = bboxes[:,:4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), 0, dtype = torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        _, ensure_pos_topk_idx = distances.topk(self.topk, dim=0, largest=False)


        # cls*iou
        cls_scores = cls_scores.sigmoid()
        decoded_bbox_pred = self.bbox_coder.decode(bboxes, bbox_preds)
        iou_pred_gt = self.iou_calculator(decoded_bbox_pred, gt_bboxes)
        
        #扩展cls_scores:
        # TODO :确定这里是原本的cls score的clone还是深copy
        cls_scores = cls_scores[:, gt_labels]
        iou_cls = iou_pred_gt * cls_scores
        iou_cls_pos = iou_cls[ensure_pos_topk_idx, range(num_gt)]
        iou_cls_thr = iou_cls_pos.mean(0) + iou_cls_pos.std(0)

        # select every lvl
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            end_idx = start_idx + bboxes_per_level
            iou_cls_per_level = iou_cls[start_idx:end_idx, :]
            selectable_k = min(self.topk*2, bboxes_per_level)
            _, topk_idxs_per_level = iou_cls_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        candidate_idxs = torch.cat((ensure_pos_topk_idx, candidate_idxs), dim=0)

        candidate_iou_cls = iou_cls[candidate_idxs, range(num_gt)]
        is_pos = candidate_iou_cls >= iou_cls_thr
        is_pos[:self.topk,:] = True


        # limit pos sample center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] = gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        later_pos = is_pos & is_in_gts


        iou_cls_inf = torch.full_like(iou_cls, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[later_pos.view(-1)]
        iou_cls_inf[index] = iou_cls.t().contiguous().view(-1)[index]
        iou_cls_inf = iou_cls_inf.view(num_gt, -1).t()
        max_iou_cls, argmax_iou_cls = iou_cls_inf.max(dim=1)
        assigned_gt_inds[max_iou_cls != -INF] = argmax_iou_cls[max_iou_cls != -INF] + 1


        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_iou_cls, labels=assigned_labels)
