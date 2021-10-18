#!/usr/bin/env bash

python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_ssdd_v6_topk5.py
python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1x_vhrvoc_v6.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6_topk5.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6_topk7.py 
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_ssdd_v6.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn_1x_vhrvoc_v6.py