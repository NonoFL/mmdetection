#!/usr/bin/env bash

python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpnv6/atss_r50_myfpnv6_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv6/atss_r50_myfpnv6_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv7/atss_r50_myfpnv7_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv7/atss_r50_myfpnv7_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv8/atss_r50_myfpnv8_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv8/atss_r50_myfpnv8_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv9/atss_r50_myfpnv9_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv9/atss_r50_myfpnv9_1x_ssdd_v6.py