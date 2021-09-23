import torch
from torch._C import ThroughputBenchmark

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ABSAssigner(BaseAssigner):

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.lamda = 0.5
        self.anchors_per_gt = 6

    def assign(self,
               points,  
               bboxes,
               cls_scores,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        '''
            Args:
            cls_scores:[num_classes, num_points]
        '''
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        overlaps = self.iou_calculator(bboxes, gt_bboxes) 

        # 不在框内的点，对应的anchor——iou值为0
        xs = points[:,0]
        ys = points[:,1]
        xs = xs[:,None]
        ys = ys[:,None]
        exp_gtbboxes = gt_bboxes[None].expand(num_bboxes, num_gts, 4)
        l_ = xs - exp_gtbboxes[...,0]
        r_ = exp_gtbboxes[...,2] - xs
        t_ = ys - exp_gtbboxes[..., 1]
        b_ = exp_gtbboxes[...,3] - ys
        distance = torch.stack((l_, t_, r_, b_), dim=-1)
        inside_gt_bbox_mask = distance.min(-1)[0] > 0
        overlaps[~inside_gt_bbox_mask] = 0

        # 每个gt_bboxes的area
        area_list = (gt_bboxes[:,2] - gt_bboxes[:, 0]) * (gt_bboxes[:,3] - gt_bboxes[:,1])

        # 对于每一个gt_bbox，计算其iou，用cls和iou的乘积
        cls_scores = cls_scores.permute(1, 0)
        scores = torch.sigmoid(cls_scores[:,gt_labels])
        
        cls_iou = torch.pow(overlaps * scores, self.lamda)
        cls_iou_mean_per_gt = self.cal_mean(cls_iou)
        cls_iou_std_per_gt = self.cal_std(cls_iou)
        cls_iou_thre_per_gt = 0.9 * cls_iou_mean_per_gt + 0.1 * cls_iou_std_per_gt

        # >cls_iou_thre_per_gt 的初步标记为pos，筛选之后肯定存在框
        is_pos = cls_iou >= cls_iou_thre_per_gt[None,:]
        cls_iou[~is_pos] = 0
        overlaps[~is_pos] = 0
        scores[~is_pos] = 0
        overlaps_copy = overlaps.clone().detach()

        # 分别计算gt框内的cls、iou的mean和std
        cls_mean_per_gt = self.cal_mean(scores)
        cls_std_per_gt = self.cal_std(scores)
        cls_thre_per_gt = 0.5 * cls_mean_per_gt + 0.5 * cls_std_per_gt
        cls_thre_per_gt = torch.clamp(cls_thre_per_gt, 0, 1)

        iou_mean_per_gt = self.cal_mean(overlaps)
        iou_std_per_gt = self.cal_std(overlaps)
        iou_thre_per_gt = 0.5 * iou_mean_per_gt + 0.5 * iou_std_per_gt
        iou_thre_per_gt = torch.clamp(iou_thre_per_gt, 0, 1)

        # 进行第二次筛选，初步设想：上一步留下的要尽可能cls或iou至少有一个大的，
        # 要注意可能存在筛选之后某个gt没有对应的anchor问题。

        # 分为3部：高标准（3），中等要求（2），低标准（1）
        # 在 is_pos上进行
        # 高标准，cls或iou其一 > 2*thre，另一 >thre(国语严格了，换成1试试)
        high_pos = ((scores >= cls_thre_per_gt[None,:]) & (overlaps >= iou_thre_per_gt[None,:])) 
        num_high_pos = sum(high_pos)
        scores[high_pos] = 0
        overlaps[high_pos] = 0          # 避免有些点既满足high和mid

        # 中标准：cls和iou > thre
        mid_pos = (scores >= cls_thre_per_gt[None,:]) | (overlaps >= iou_thre_per_gt[None,:])
        num_mid_pos = sum(mid_pos)
        scores[mid_pos] = 0
        overlaps[mid_pos] = 0
        
        # 低标准：
        is_pos[high_pos] = False
        is_pos[mid_pos] = False

        # 低标准：is_pos

        # 确定anchor：
        fix_anchor_per_gt = torch.full((num_gts,), self.anchors_per_gt)

        '''如果high够，排序，选最大的，mid够，排序。。。high和mid，low有关系？？怎么排序？'''
        # high好说，score + ovelap？？或者和mid同？？， high应反映综合的，用pow(,0.5)
        # mid 也先用pow（）试试daisan
       
        # 总思路，high，mid依次从高排序，再截断
        high_value_list, high_id_list = self.split_and_sort(overlaps_copy, high_pos)
        mid_value_list, mid_id_list = self.split_and_sort(overlaps_copy, mid_pos)
        low_value_list, low_id_list = self.split_and_sort(overlaps_copy, is_pos)

        # num_high_pos, mid,
        pos_id = torch.zeros_like(overlaps).type_as(overlaps)
        for i in range(num_gts): 
            if num_high_pos[i] >= self.anchors_per_gt:
                pos_id[high_id_list[i][:self.anchors_per_gt],i] = 1
            elif num_mid_pos[i] + num_high_pos[i] >= self.anchors_per_gt:
                pos_id[high_id_list[i], i] = 1
                pos_id[mid_id_list[i][:self.anchors_per_gt-num_high_pos[i]], i] = 1
            else:
                pos_id[high_id_list[i],i] = 1
                pos_id[mid_id_list[i], i] = 1
                pos_id[low_id_list[i][:self.anchors_per_gt-num_high_pos[i]-num_mid_pos[i]],1] = 1
        pos_id = torch.where(pos_id==1, True, False) 

        max_overlaps = torch.ones((num_bboxes,)).type_as(overlaps).type(torch.long)
        max_overlaps = max_overlaps * (-INF)
        assigned_gt_inds = torch.zeros((num_bboxes,)).type_as(overlaps).type(torch.long)
        if gt_labels is not None:
            assigned_labels = torch.ones((num_bboxes,)).type_as(overlaps).type(torch.long)
            assigned_labels = assigned_labels * (-1)
            for i in range(num_gts):
                assigned_labels[pos_id[:,i]] = int(gt_labels[i])
                assigned_gt_inds[pos_id[:,i]] = i+1
                max_overlaps[pos_id[:,i]] = (overlaps_copy[pos_id[:,i],i]).type(torch.long)

        else:
            assigned_labels = None
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, assigned_labels)

        
        # 若一个anchor被多个gt选择，则选择xxxx

    def cal_mean(self, tens):
        lamda = len(tens) / sum(tens > 0)
        return tens.mean(0) * lamda
    
    def cal_std(self, tens):
        lamda = (len(tens)-1) / (sum(tens > 0)-1)
        lamda = torch.pow(lamda, 1/2)
        return tens.std(0) * lamda

    def split_and_sort(self,tens,tens_pos):
        num_tens = sum(tens_pos)    #[1,2,3,41]
        # 找出tens_pos>0 的坐标
        pos_list = []
        pos_list_indice = []
        num_gt = len(num_tens)        #gt 的数量
        for i in range(num_gt):
            ten_pos = tens_pos[:,i]
            ten = tens[:,i]
            if num_tens[i] == 0:
                sorted_pos_score = []
                sorted_pos_score_id = []
            pos_score = ten[ten_pos]
            pos_score_id = torch.nonzero(ten_pos>0)
            sorted_pos_score, sorted_pos_indice = torch.sort(pos_score,descending=True)
            sorted_pos_score_id = pos_score_id[sorted_pos_indice]
            pos_list.append(sorted_pos_score)
            pos_list_indice.append(sorted_pos_score_id)
        return pos_list, pos_list_indice

