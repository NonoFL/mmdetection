# dataset settings
dataset_type = "SsddDataset"
data_root = "data/SSDD_coco/"
img_norm_cfg = dict(mean=[35.757, 39.758, 39.752], std=[27.732, 27.734, 27.718], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,      #batchsize
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        img_prefix=data_root + "train/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "test/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "test/",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")    #interval:更改val的间隔
