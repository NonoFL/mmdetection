#!/usr/bin/env bash

python tools/train.py configs/gflv2/gflv2_r50_fpn_1x_vhrvoc.py
python tools/train.py configs/gflv2/gflv2_r50_fpn_1x_ssdd.py
