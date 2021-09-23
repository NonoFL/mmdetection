#!/usr/bin/env bash

python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_centersampling_norm-on-bbox_ssdd.py
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_centersampling_norm-on-bbox_vhrvoc.py
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_centersampling_ssdd.py
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_centersampling_vhrvoc.py
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_norm-on-bbox_ssdd.py
python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_norm-on-bbox_vhrvoc.py