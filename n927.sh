#!/usr/bin/env bash

python tools/train.py configs/atss/atss_r50_fpn_1x_vhrvoc_v4.py
python tools/train.py configs/atss/atss_r50_fpn_1x_ssdd_v4.py
