#!/usr/bin/env bash

python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk50.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck9_topk50.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk15.py
python tools/train.py configs/atss/atss_assignv9/atss_r50_fpn_1x_ssdd_v9_ck7_topk17.py
