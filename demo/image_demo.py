import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/atss_r50_fpn_1x_vhrvoc.py work_dirs/atss/ATSS/atss_r50_fpn_1x_vhrvoc/epoch_12.pth --device cpu
# python demo/image_demo.py data/VHR_voc/JPEGImages/006.jpg work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/atss_r50_myfpn-1_1x_vhrvoc_v6.py work_dirs/atss/ATSS_myfpn/atss_r50_myfpn-1_1x_vhrvoc_v6（1）/epoch_11.pth --device cpu
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result,_= inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)
''''''''

if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
