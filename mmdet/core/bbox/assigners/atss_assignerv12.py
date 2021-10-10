import math
import torch
import random
from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ATSSAssignerv12(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 overlapEPS=0.025,
                 distanceEPS=16,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.overlapsEPS = overlapEPS
        self.distanceEPS = distanceEPS
        # self.dbscan = DBSCAN(eps=0.5, min_samples=5)

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):

        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        gt_bboxes = gt_bboxes[None].expand(num_bboxes, num_gt, 4)
        xs, ys = bboxes_points[:, 0], bboxes_points[:, 1]
        xs = xs[:, None]
        ys = ys[:, None]
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        is_in_gt = torch.stack((left, top, right, bottom), -1).min(dim=-1)[0] > 0.01
        print('--------------------------------------------------------------')
        print("sum(is_in_gt):  " + str(sum(is_in_gt)))


        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

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

        thresh_list = []
        for gt in range(num_gt):
            is_in_gt_alone = is_in_gt[:, gt]
            overlaps_alone = overlaps[:, gt]
            point_list = []
            overlaps_list = []
            for i in range(num_bboxes):
                if is_in_gt_alone[i,] == True:
                    point_list.append(bboxes_points[i,])
                    overlaps_list.append(overlaps_alone[i])

            if len(point_list) > 200:       # 如果一个gt内bbox太多，用DBSCAN耗时间严重，采用一刀切
                overlap_list_tensor = torch.tensor(overlaps_list)
                # 选择top80
                sorted_overlap_list_tensor, _ = torch.sort(overlap_list_tensor, descending=True)
                thresh_temp = 0.4 if sorted_overlap_list_tensor[30] > 0.4 else sorted_overlap_list_tensor[40]
                thresh_list.append(thresh_temp)
            else:
                gama = [x for x in range(len(point_list))]
                cluster = [-1 for y in range(len(point_list))]
                k = -1
                fil = []
                while(len(gama) > 0):
                    point_k = random.choice(gama)
                    gama.remove(point_k)
                    fil.append(point_k)

                    NeighborPoints = findNeiborPoint(point_k, point_list, overlaps_list, self.distanceEPS, self.overlapsEPS)

                    if len(NeighborPoints) == 1:
                        cluster[point_k] = -1   # 周边没有满足条件的, 标记为-1
                    else:
                        k += 1
                        cluster[point_k] = k
                        for i in NeighborPoints:
                            if i not in fil:
                                gama.remove(i)
                                fil.append(i)
                                Ner_Neighborpts = findNeiborPoint(i, point_list, overlaps_list, self.distanceEPS, self.overlapsEPS)
                                if len(Ner_Neighborpts) > 1:
                                    for j in Ner_Neighborpts:
                                        if j not in NeighborPoints:
                                            NeighborPoints.append(j)
                                if cluster[i] == -1:
                                    cluster[i] = k

                # if k>-1:
                kk =  math.ceil(k / 2 + 1)      #k2,kk2,(0,1,2)
                overlaps_per_cluster_mean = []
                overlaps_per_cluster_std = []
                c = torch.tensor(cluster)
                overlaps_c = torch.tensor(overlaps_list)
                for ii in range(-1,k+1):
                    mean = overlaps_c[c==ii].mean()
                    std = overlaps_c[c==ii].std()
                    overlaps_per_cluster_mean.append(mean)
                    overlaps_per_cluster_std.append(std)

                overlaps_per_cluster_mean = torch.tensor(overlaps_per_cluster_mean)
                overlaps_per_cluster_std = torch.tensor(overlaps_per_cluster_std)

                chakanmean, indices_index = torch.sort(overlaps_per_cluster_mean, descending=True)
                print(chakanmean[:kk])
                # indices_index = indices_index - 1 #更符合聚类名称

                flag_candidata = (c==indices_index[0]-1)
                for n in range(1, kk):
                    # stdvalue = overlaps_per_cluster_std[indices_index[n-1]] if overlaps_per_cluster_std[indices_index[n-1]] > 0 else 0.3*overlaps_per_cluster_mean[indices_index[n-1]]
                    stdvalue = 0.3*overlaps_per_cluster_mean[indices_index[n-1]]
                    if overlaps_per_cluster_mean[indices_index[n]] >= overlaps_per_cluster_mean[indices_index[n-1]] - stdvalue:
                        if kk>3 & n>=3:
                            # if (overlaps_per_cluster_mean[indices_index[0]] - overlaps_per_cluster_mean[indices_index[n]] < 0.15) & (overlaps_per_cluster_mean[indices_index[0]]<0.4):
                            if overlaps_per_cluster_mean[indices_index[n]] < torch.tensor([0.2]):
                                break
                            elif kk>5: 
                                if overlaps_per_cluster_mean[indices_index[n]]>torch.tensor([0.35]):
                                    flag_candidata += (c==indices_index[n]-1)
                                else:
                                    break
                            else:
                                break
                                
                        else:
                            flag_candidata += (c==indices_index[n]-1)
                    else:
                        break

                candidate_overlaps = overlaps_c[flag_candidata==True]
                mean, std = candidate_overlaps.mean(),candidate_overlaps.std()
                thresh = mean - std         #采用mean-std的方式，可尝试mean+std的效果做对比。
                thresh_list.append(thresh)
                # else:
                #     #选择最大的9个数
                #     pass

        thresh = torch.tensor(thresh_list).type_as(overlaps)
        thresh = thresh.to(overlaps.device)
        print(thresh)
        candidata = overlaps>=thresh
        candidata = candidata & is_in_gt
        print(sum(candidata)) 

        overlaps_inf = torch.full_like(overlaps,-INF)
        overlaps_inf[candidata==True] = overlaps[candidata==True]

        # 如果


        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

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
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        


def findNeiborPoint(k, point_list, overlaps_list, distanceEPS,overlapsEPS):
    N = []
    O = []
    for p in range(len(point_list)):
        temp_d = (point_list[k]-point_list[p]).pow(2).sum(-1).sqrt()
        temp_o = torch.abs(overlaps_list[k]-overlaps_list[p])
        if (temp_d < distanceEPS) & (temp_o < overlapsEPS):
            N.append(p)

    return N