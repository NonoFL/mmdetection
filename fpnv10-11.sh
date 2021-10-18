#!/usr/bin/env bash

python tools/train.py configs/atss/myfpnv10/atss_r50_myfpnv10_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv10/atss_r50_myfpnv10_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpnv11/atss_r50_myfpnv11_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv11/atss_r50_myfpnv11_1x_vhrvoc_v6.py