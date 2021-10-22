#!/usr/bin/env bash

python tools/train.py configs/atss/myfpnv12/atss_r50_myfpnv12_1_8_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv12/atss_r50_myfpnv12_1_8_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv12/atss_r50_myfpnv12_1_9_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv12/atss_r50_myfpnv12_1_9_1x_vhrvoc_v6.py