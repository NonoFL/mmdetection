# dataset settings
dataset_type = 'VhrvocDataset'
data_root = 'data/VHR_voc/'
img_norm_cfg = dict(mean=[86.218, 91.479, 81.957], std=[41.654, 39.424, 38.359], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'ImageSets/Main/train.txt',
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
