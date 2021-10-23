# from mmcv.runner import checkpoint
# from mmdet.apis import inference_detector, init_detector
# import cv2
# import numpy as np
# import time
# import torch
# import os

# def main():

#     # config = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/atss_r50_myfpn-1_1x_vhrvoc_v6.py'
#     # checkpoint = 'work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/epoch_11.pth'
#     config = 'work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py'
#     checkpoint = 'work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_12.pth'
#     device = 'cuda:0'
#     # build the model from a config file and a checkpoint file
#     model = init_detector(config, checkpoint, device=device)
#     # test a single image
#     img = 'data/VHR_voc/JPEGImages/006.jpg'
#     image = cv2.imread(img)
#     height, width, channels = image.shape
#     result, x_backone, x_fpn = inference_detector(model, img)

#     if not os.path.exists('demo/feature_map'):
#         os.makedirs('demo/feature_map')

#     feature_index = 0
#     for feature in x_backone:
#         feature_index += 1
#         P = torch.sigmoid(feature)
#         P = P.cpu().detach().numpy()
#         P = np.maximum(P, 0)
#         P = (P - np.min(P)) / (np.max(P) - np.min(P))
#         P = P.squeeze(0)
#         print(P.shape)

#         P = P[10, ...]  # 挑选一个通道
#         print(P.shape)

#         cam = cv2.resize(P, (width, height))
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255 
#         heatmap = heatmap / np.max(heatmap)
#         heatmap_image = np.uint8(255 * heatmap)

#         cv2.imwrite('demo/feature_map/' + 'stage_' + str(feature_index) + '_heatmap.jpg', heatmap_image)
#         result = cv2.addWeighted(image, 0.8, heatmap_image, 0.3, 0)
#         cv2.imwrite('demo/feature_map/' + 'stage_' + str(feature_index) + '_result.jpg', result)

#     feature_index = 1
#     for feature in x_fpn:
#         feature_index += 1
#         P = torch.sigmoid(feature)
#         P = P.cpu().detach().numpy()
#         P = np.maximum(P, 0)
#         P = (P - np.min(P)) / (np.max(P) - np.min(P))
#         P = P.squeeze(0)
#         P = P[2, ...]
#         print(P.shape)
#         cam = cv2.resize(P, (width, height))
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255
#         heatmap = heatmap / np.max(heatmap)
#         heatmap_image = np.uint8(255 * heatmap)

#         cv2.imwrite('demo/feature_map/' + 'P' + str(feature_index) + '_heatmap.jpg', heatmap_image)  # 生成图像
#         result = cv2.addWeighted(image, 0.8, heatmap_image, 0.4, 0)
#         cv2.imwrite('demo/feature_map/' + 'P' + str(feature_index) + '_result.jpg', result)


# if __name__ == '__main__':
#     main()

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


#coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def draw_feature_map(model, img_path, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = mmcv.imread(img_path)
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    result, featuremaps = inference_detector(model, img) #这里需要改model，让其在forward的最后return特征图。我这里return的是一个Tensor的tuple，每个Tensor对应一个level上输出的特征图。
    i=0
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.5 + img*0.3  # 这里的0.4是热力图强度因子
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(save_dir,'featuremap_'+str(i)+'.png'), superimposed_img)  # 将图像保存到硬盘
        i=i+1
    show_result_pyplot(model, img, result, score_thr=0.05)


from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--save_dir', help='Dir to save heatmap image',default='demo/feature_map/')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    draw_feature_map(model,args.img,args.save_dir)

if __name__ == '__main__':
    main()