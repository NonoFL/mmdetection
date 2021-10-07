#!/usr/bin/env bash

python tools/train.py configs/atss/atss_r50_myfpn_1x_ssdd_v6.py
python tools/train.py configs/atss/atss_r50_myfpn_1x_vhrvoc_v6.py
python tools/train.py configs/atss/atss_r50_myfpn_1x_vhrvoc_v6_lrchange.py