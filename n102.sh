#!/usr/bin/env bash

python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v6_topk11.py
python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v6_topk13.py
python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v7_topk13.py


