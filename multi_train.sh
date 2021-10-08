#!/usr/bin/env bash

# python tools/train.py configs/atss/atss_r50_myfpn_1x_ssdd_v6.py
# python tools/train.py configs/atss/atss_r50_myfpn_1x_vhrvoc_v6.py

python tools/train.py configs/atss/myfpn/atss_r50_myfpn_1x_vhrvoc_v6_lrchange.py

#FPN-1
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn-1_1x_ssdd_v6.py
python tools/train.py configs/atss/atss_assignv6/atss_r50_fpn-1_1x_vhrvoc_v6.py

#MyFPNv4-1
python tools/train.py configs/atss/myfpnv4/atss_r50_myfpnv4-1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv4/atss_r50_myfpnv4-1_1x_vhrvoc_v6.py
#MyFPNv3-1
python tools/train.py  configs/atss/myfpnv3/atss_r50_myfpnv3-1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpnv3/atss_r50_myfpnv3-1_1x_vhrvoc_v6.py
#MyFPNv2-1 
python tools/train.py configs/atss/myfpnv2/atss_r50_myfpnv2_1x_vhrvoc_v6.py
python tools/train.py configs/atss/myfpnv2/atss_r50_myfpnv2-1_1x_ssdd_v6.py
#MyFPN-1 
python tools/train.py configs/atss/myfpn/atss_r50_myfpn-1_1x_ssdd_v6.py
python tools/train.py configs/atss/myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6.py

