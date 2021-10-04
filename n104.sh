#!/usr/bin/env bash

python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v8.py
python tools/train.py configs/atss/atss_r50_fpn_1x_vhrvoc_v8.py
python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v8_topk11.py
python tools/train.py configs/atss/atss_r50_fpn_1x_vhrvoc_v8_topk11.py
python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v8_topk13.py
python tools/train.py configs/atss/atss_r50_fpn_1x_vhrvoc_v8_topk13.py
