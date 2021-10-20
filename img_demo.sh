#!/bin/sh

# Folder_A="/home/apple/File/Study/C/GIT/mmdetection/data/VHR_voc/JPEGImages/"



# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/VHR/0.5/ATSS_res/" $Folder_A$temp_file work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_14.pth
# done

# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/VHR/0.5/ATSS_v6_res/" $Folder_A$temp_file work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/atss_r50_fpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_vhrvoc_v6/epoch_18.pth
# done


# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/VHR/0.5/ATSS_v6_myfpn_res/" $Folder_A$temp_file work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/atss_r50_myfpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/epoch_9.pth
# done

# Folder_A="/home/apple/File/Study/C/GIT/mmdetection/data/SSDD_voc/JPEGImages/"


# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/SSDD/0.5/ATSS_res/" $Folder_A$temp_file /home/apple/File/Study/C/GIT/mmdetection/work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/atss_r50_fpn_1x_ssdd.py work_dirs/atss/ATSS/atss_r50_fpn_1x_ssdd/epoch_80.pth
# done

# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/SSDD/0.5/ATSS_v6_res/" $Folder_A$temp_file work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/atss_r50_fpn_1x_ssdd_v6.py work_dirs/atss/ATSS_v6/atss_r50_fpn_1x_ssdd_v6/epoch_64.pth
# done


# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/SSDD/0.5/ATSS_v6_myfpn_res/" $Folder_A$temp_file work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/atss_r50_myfpn_1x_ssdd_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_ssdd_v6/epoch_62.pth
# done



Folder_A="/home/apple/File/Study/C/GIT/mmdetection/data/VHR_voc/JPEGImages/"

# for file in ${Folder_A}/*
# do 
#     temp_file=`basename $file`
#     echo $Folder_A$temp_file

#     python demo/image_demo.py "demo/VHR/0.5/ATSS_v6_myfpnv12_res/" $Folder_A$temp_file work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/atss_r50_myfpnv12_1_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpnv12/atss_r50_myfpnv12_1_1x_vhrvoc_v6/epoch_9.pth
# done


for file in ${Folder_A}/*
do 
    temp_file=`basename $file`
    echo $Folder_A$temp_file

    python demo/image_demo.py "demo/VHR/0.3/ATSS_v6_myfpn_res/" $Folder_A$temp_file work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/atss_r50_myfpn_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn_1x_vhrvoc_v6_1/epoch_9.pth
done
