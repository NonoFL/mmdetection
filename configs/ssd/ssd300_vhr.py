_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/vhr_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'SsddDataset'
data_root = 'data/SSDD_voc/'
img_norm_cfg = dict(mean=[86.218, 91.479, 81.957], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/train.json',
            img_prefix=data_root + 'train/',
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]