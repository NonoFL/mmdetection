from numpy.core.numeric import indices
import torch
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from sklearn import datasets

distanceEPS = 16
overlapsEPS = 0.05

def caldistance(a,b):
    return (a[:,None] - b[None,:]).pow(2).sum(-1).sqrt()
def caloverlaps(a,b):
    return a-b

def findNeiborPoint(k, X, eps, overlaps_list):
    N = []
    O = []
    for p in range(len(X)):
        temp_d = (X[k]-X[p]).pow(2).sum(-1).sqrt()
        temp_o = torch.abs(overlaps_list[k]-overlaps_list[p])
        if (temp_d < eps) & (temp_o < overlapsEPS):
            N.append(p)

    return N



overlaps = np.loadtxt('/home/apple/File/Study/C/GIT/mmdetection/overlaps.txt')
distances = np.loadtxt('/home/apple/File/Study/C/GIT/mmdetection/distance.txt')
bboxes_points = np.loadtxt('/home/apple/File/Study/C/GIT/mmdetection/bboxes_points.txt')
gt_bboxes = np.loadtxt('/home/apple/File/Study/C/GIT/mmdetection/gt_bboxes.txt')
overlaps = torch.from_numpy(overlaps)
distances = torch.from_numpy(distances)
bboxes_points = torch.from_numpy(bboxes_points)
gt_bboxes = torch.from_numpy(gt_bboxes)
num_classes = 10

num_points, num_gts = distances.size(0), distances.size(1)
# overlapsbig0 = overlaps>0

# flag_visited = torch.zeros_like(distances)
# flag_temp = torch.zeros_like(distances)
# flag_class = torch.full_like(distances, 0)  #集成时为-1

# gt_bboxes_alone = gt_bboxes.clone()
gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
xs, ys = bboxes_points[:, 0], bboxes_points[:, 1]
xs = xs[:, None]
ys = ys[:, None]
left = xs - gt_bboxes[..., 0]
right = gt_bboxes[..., 2] - xs
top = ys - gt_bboxes[..., 1]
bottom = gt_bboxes[..., 3] - ys
is_in_gt = torch.stack((left, top, right, bottom), -1).min(dim=-1)[0] > 0.01 
thresh_list = []
for gt in range(num_gts):
    is_in_gt_alone = is_in_gt[:, gt]
    overlaps_alone = overlaps[:, gt]
    point_list = []
    overlaps_list = []
    for i in range(num_points):
        if is_in_gt_alone[i,] == True:
            point_list.append(bboxes_points[i,])
            overlaps_list.append(overlaps_alone[i])
    visited_list = []
    gama = [x for x in range(len(point_list))]
    cluster = [-1 for y in range(len(point_list))]
    k = -1

    fil = []

    while(len(gama) > 0):
        point_k = random.choice(gama)
        gama.remove(point_k)
        fil.append(point_k)


        NeighborPoints = findNeiborPoint(point_k, point_list, distanceEPS, overlaps_list)

        if len(NeighborPoints) == 1:
            cluster[point_k] = -1   # 周边没有满足条件的, 标记为-1
        else:
            k += 1
            cluster[point_k] = k
            for i in NeighborPoints:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_Neighborpts = findNeiborPoint(i, point_list, distanceEPS, overlaps_list)
                    if len(Ner_Neighborpts) > 1:
                        for j in Ner_Neighborpts:
                            if j not in NeighborPoints:
                                NeighborPoints.append(j)
                    if cluster[i] == -1:
                        cluster[i] = k
    print("--------")
    print(cluster)
    print(overlaps_list)
    print("--------")
    

    # 共分成了k类，选取50%的聚类(上取整)，计算其总共的mean+std，>的，为正, 可尝试只选择一类，然后把这类的所有当作正，其余不管
    
    # 需要修正：计算每一类的均值，从大到小排序，3sigma原则
    
    
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

    _, indices_index = torch.sort(overlaps_per_cluster_mean, descending=True)
    # indices_index = indices_index - 1 #更符合聚类名称

    flag_candidata = (c==indices_index[0]-1)
    for n in range(1, kk):
        # stdvalue = overlaps_per_cluster_std[indices_index[n-1]] if overlaps_per_cluster_std[indices_index[n-1]] > 0 else 0.3*overlaps_per_cluster_mean[indices_index[n-1]]
        stdvalue = 0.3*overlaps_per_cluster_mean[indices_index[n-1]]

        if overlaps_per_cluster_mean[indices_index[n]] >= overlaps_per_cluster_mean[indices_index[n-1]] - stdvalue:
            flag_candidata += (c==indices_index[n]-1)
        else:
            n ==kk

    candidate_overlaps = overlaps_c[flag_candidata==True]
    mean, std = candidate_overlaps.mean(),candidate_overlaps.std()
    thresh = mean - std         #采用mean-std的方式，可尝试mean+std的效果做对比。
    thresh_list.append(thresh)
    
# 切回到最开始有num_points点的overlap
thresh = torch.tensor(thresh_list)
print(thresh)
candidata = overlaps>=thresh



    







    # plt.figure(figsize=(12,12),dpi=80)
    # P = []
    # for i in point_list:
    #     i = i.tolist()
    #     P.append(i)
    # P = torch.tensor(P)
    # plt.scatter(P[:,0], P[:, 1], c=cluster)
    # plt.show()
    # print("----------")


