#!/usr/bin/env bash

# python tools/train.py configs/atss/atss_r50_myfpnv3_1x_ssdd_v6.py
# python tools/train.py configs/atss/atss_r50_myfpnv3_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpnv5/atss_r50_myfpnv5_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv5/atss_r50_myfpnv5_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv5/atss_r50_myfpnv5-1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv5/atss_r50_myfpnv5-1_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpnv4/atss_r50_myfpnv4-1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv4/atss_r50_myfpnv4-1_1x_vhrvoc_v6.py