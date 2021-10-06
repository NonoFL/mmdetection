from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os

def main():

    config = 'work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py'
    checkpoint = 'work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_12.pth'
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    img = 'vhr/279.jpg'
    image = cv2.imread(img)
    height, width, channels = image.shape
    result, x_fpn = inference_detector(model, img)

    if not os.path.exists('feature_map'):
        os.makedirs('feature_map')

    # feature_index = 0
    # for feature in x_backone:
    #     feature_index += 1
    #     P = torch.sigmoid(feature)
    #     P = P.cpu().detach().numpy()
    #     P = np.maximum(P, 0)
    #     P = (P - np.min(P)) / (np.max(P) - np.min(P))
    #     P = P.squeeze(0)
    #     print(P.shape)

    #     P = P[10, ...]  # 挑选一个通道
    #     print(P.shape)

    #     cam = cv2.resize(P, (width, height))
    #     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     heatmap = heatmap / np.max(heatmap)
    #     heatmap_image = np.uint8(255 * heatmap)

    #     cv2.imwrite('feature_map/' + 'stage_' + str(feature_index) + '_heatmap.jpg', heatmap_image)
    #     result = cv2.addWeighted(image, 0.8, heatmap_image, 0.3, 0)
    #     cv2.imwrite('feature_map/' + 'stage_' + str(feature_index) + '_result.jpg', result)

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

        cv2.imwrite('feature_map/' + 'P' + str(feature_index) + '_heatmap.jpg', heatmap_image)  # 生成图像
        result = cv2.addWeighted(image, 0.8, heatmap_image, 0.4, 0)
        cv2.imwrite('feature_map/' + 'P' + str(feature_index) + '_result.jpg', result)


if __name__ == '__main__':
    main()