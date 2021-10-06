#!/usr/bin/env bash

python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk5.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk9.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk11.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk13.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck9_topk11.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck9_topk13.py

python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_vhrvoc_v9_ck7_topk9.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_vhrvoc_v9_ck7_topk11.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_vhrvoc_v9_ck7_topk13.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_vhrvoc_v9_ck9_topk11.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_vhrvoc_v9_ck9_topk13.py