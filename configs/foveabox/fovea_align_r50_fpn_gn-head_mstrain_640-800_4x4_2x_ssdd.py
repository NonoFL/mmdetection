_base_ = './fovea_r50_fpn_4x4_1x_ssdd.py'
model = dict(
    bbox_head=dict(
        with_deform=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
img_norm_cfg = dict(
    mean=[35.757, 39.758, 39.752], std=[27.732, 27.734, 27.718], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))
# learning policy

lr_config = dict(step=[60,95])
runner = dict(type='EpochBasedRunner', max_epochs=100)
