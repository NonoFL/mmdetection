#!/usr/bin/env bash


# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_12.pth
# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth
# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/atss_r50_myfpn-1_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/epoch_11.pth --device cpu
# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS_v12/atss_r50_fpn_1x_vhrvoc_v12/atss_r50_fpn_1x_vhrvoc_v12.py work_dirs/atss/ATSS_v12/atss_r50_fpn_1x_vhrvoc_v12/epoch_7.pth

# python demo/image_demo.py data/SSDD_voc/JPEGImages/000003.jpg work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/atss_r50_fpn_1x_ssdd.py work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/epoch_80.pth
# python demo/image_demo.py data/SSDD_voc/JPEGImages/000003.jpg work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/atss_r50_fpn_1x_ssdd_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/epoch_66.pth
# python demo/image_demo.py data/SSDD_voc/JPEGImages/000003.jpg work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/atss_r50_myfpn_1x_ssdd_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/epoch_62.pth
# python demo/image_demo.py data/SSDD_voc/JPEGImages/000003.jpg work_dirs/atss/ATSS_v12/atss_r50_fpn_1x_vhrvoc_v12/atss_r50_fpn_1x_vhrvoc_v12.py work_dirs/atss/ATSS_v12/atss_r50_fpn_1x_vhrvoc_v12/epoch_7.pth

python demo/heatmap_visual.py data/VHR_voc/JPEGImages/176.jpg work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_12.pth --save_dir demo/feature_map/atss/
python demo/heatmap_visual.py data/VHR_voc/JPEGImages/176.jpg work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth --save_dir demo/feature_map/atss_v6/
python demo/heatmap_visual.py data/VHR_voc/JPEGImages/176.jpg work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/atss_r50_myfpn-1_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/epoch_11.pth  --save_dir demo/feature_map/atss_myfpn-1/
python demo/heatmap_visual.py data/VHR_voc/JPEGImages/176.jpg work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/atss_r50_myfpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/epoch_9.pth --save_dir demo/feature_map/atss_myfpn/
python demo/heatmap_visual.py data/VHR_voc/JPEGImages/176.jpg work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/atss_r50_myfpnv12_1_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/epoch_9.pth --save_dir demo/feature_map/atss_myfpn12_1/