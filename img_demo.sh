#!/bin/sh

Folder_A="/home/apple/File/Study/C/GIT/mmdetection/data/VHR_voc/JPEGImages/"



for file in ${Folder_A}/*
do 
    temp_file=`basename $file`
    echo $Folder_A$temp_file

    python demo/image_demo.py "demo/VHR/ATSS_v6_res/" $Folder_A$temp_file work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth
done

for file in ${Folder_A}/*
do 
    temp_file=`basename $file`
    echo $Folder_A$temp_file

    python demo/image_demo.py "demo/VHR/0.5/ATSS_res/" $Folder_A$temp_file work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_14.pth
done


for file in ${Folder_A}/*
do 
    temp_file=`basename $file`
    echo $Folder_A$temp_file

    python demo/image_demo.py "demo/VHR/0.5/ATSS_v6_res/" $Folder_A$temp_file work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth
done

