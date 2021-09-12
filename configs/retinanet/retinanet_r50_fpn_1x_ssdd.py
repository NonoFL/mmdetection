_base_ = [
    '../_base_/models/retinanet_r50_fpn_ssdd.py',
    '../_base_/datasets/ssdd_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
