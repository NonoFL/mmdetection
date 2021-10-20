from mmcv.runner import checkpoint
from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os

def main():

    # config = 'work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/atss_r50_fpn_1x_ssdd_v6.py'
    # checkpoint = 'work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/epoch_64.pth'

    # config = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/atss_r50_myfpn_1x_ssdd_v6.py'
    # checkpoint = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/epoch_62.pth'

    # checkpoint = 'work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth' 
    # config = 'work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py'

    # config = "work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/atss_r50_myfpnv12_1_1x_vhrvoc_v6.py" 
    # checkpoint = "work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/epoch_9.pth"

    # checkpoint = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/epoch_11.pth' 
    # config = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/atss_r50_myfpn-1_1x_vhrvoc_v6.py'

    config = "work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/atss_r50_fpn_1x_ssdd.py" 
    checkpoint = "work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/epoch_80.pth"


    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    # img = 'demo/ssdd/000697.jpg'
    img = 'data/VHR_voc/JPEGImages/176.jpg'
    image = cv2.imread(img)
    height, width, channels = image.shape
    result, x_backone, x_fpn = inference_detector(model, img)

    if not os.path.exists('demo/feature_map'):
        os.makedirs('demo/feature_map')

    feature_index = 0
    for feature in x_backone:
        feature_index += 1
        P = torch.sigmoid(feature)
        P = P.cpu().detach().numpy()
        P = np.maximum(P, 0)
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        P = P.squeeze(0)
        print(P.shape)

        P = P[10, ...]  # 挑选一个通道
        print(P.shape)

        cam = cv2.resize(P, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255 
        heatmap = heatmap / np.max(heatmap)
        heatmap_image = np.uint8(255 * heatmap)

        cv2.imwrite('demo/feature_map/' + 'stage_' + str(feature_index) + '_heatmap.jpg', heatmap_image)
        result = cv2.addWeighted(image, 0.8, heatmap_image, 0.3, 0)
        cv2.imwrite('demo/feature_map/' + 'stage_' + str(feature_index) + '_result.jpg', result)

    feature_index = 1
    for feature in x_fpn:
        feature_index += 1
        P = torch.sigmoid(feature)
        P = P.cpu().detach().numpy()
        P = np.maximum(P, 0)
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        P = P.squeeze(0)
        P = P[2, ...]
        print(P.shape)
        cam = cv2.resize(P, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        heatmap_image = np.uint8(255 * heatmap)

        cv2.imwrite('demo/feature_map/' + 'P' + str(feature_index) + '_heatmap.jpg', heatmap_image)  # 生成图像
        result = cv2.addWeighted(image, 0.8, heatmap_image, 0.4, 0)
        cv2.imwrite('demo/feature_map/' + 'P' + str(feature_index) + '_result.jpg', result)


if __name__ == '__main__':
    main()

'''
if you use two-stage detector, such as faster rcnn,please change the codes :
1. mmdet/models/detectors/two_stage.py
    def extract_feat(self, img):
    """Directly extract features from the backbone+neck
    """     
    x_backbone = self.backbone(img)
    if self.with_neck:
        x_fpn = self.neck(x_backbone)
    return x_backbone,x_fpn
and:
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x_backbone,x_fpn = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.simple_test_rpn(x_fpn, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(
            x_fpn, proposal_list, img_metas, rescale=rescale),x_backbone,x_fpn
2.mmdet/apis/inference.py
    def inference_detector(model, img):
    .......
            # forward the model
        with torch.no_grad():
            result,x_backbone,x_fpn= model(return_loss=False, rescale=True, **data)
        return result,x_backbone,x_fpn
if you use other detectors, it is easy to achieve it like this
'''