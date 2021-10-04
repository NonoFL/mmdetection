#!/usr/bin/env bash

python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6_topk7.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_ssdd_v6_topk7.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6_topk8.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_ssdd_v6_topk8.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6_topk5.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_ssdd_v6_topk5.py

python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_ssdd_v4_topk7.py
python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_ssdd_v4_topk11.py
python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_ssdd_v4_topk13.py
python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_vhrvoc_v4_topk7.py
python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_vhrvoc_v4_topk11.py
python tools/train.py configs/atss/atss_assignv4/atss_r50_fpn_1x_vhrvoc_v4_topk13.py