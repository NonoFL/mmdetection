# dataset settings
dataset_type = "VhrDataset"
data_root = "data/VHR_coco/"
img_norm_cfg = dict(mean=[86.218, 91.479, 81.957], std=[41.654, 39.424, 38.359], to_rgb=True)
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
    samples_per_gpu=2,
    workers_per_gpu=2,
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
evaluation = dict(interval=1, metric="bbox")
