import cv2
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from datasets import get_dataset_test
from datasets.transforms import ToTensor
from spn_train import SuperPixelNet
from utils import get_superpixel_map


def get_input_size(input_s: int, image_s: int):
    if input_s is not None:
        assert input_s % 16 == 0, 'input size should be a multiple of 16'
        return input_s
    else:
        return (image_s // 16) * 16


def main(args):
    spn_model = SuperPixelNet.load_from_checkpoint(args.model_path, map_location='cpu')
    spn_model.eval().cuda()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = spn_model.hparams.dataset
    downsize = spn_model.hparams.downsize
    if args.data_dir is None:
        args.data_dir = spn_model.hparams.data_dir

    ds = get_dataset_test(dataset_name, args.data_dir, ToTensor())

    bar = tqdm(range((len(ds))))
    for i in bar:
        name_id = ds.index[i]

        img, _ = ds[i]
        img = torch.from_numpy(img).cuda().unsqueeze(0)

        ori_h, ori_w = img.shape[-2:]

        input_h = get_input_size(args.input_image_height, ori_h)
        input_w = get_input_size(args.input_image_width, ori_w)

        if ori_h != input_h or ori_w != input_w:
            img = F.interpolate(img, [input_h, input_w], mode='bicubic', align_corners=True)

        with torch.no_grad():
            pred_q, _ = spn_model(img)

        superpixel_map = get_superpixel_map(input_h, input_w, downsize, 1, pred_q, pred_q.device)

        if ori_h != input_h or ori_w != input_w:
            superpixel_map = F.interpolate(superpixel_map.float(), [ori_h, ori_w], mode='nearest').long()

        sp_map_np = superpixel_map.squeeze().cpu().numpy()

        # save superpixel map
        img = ds.read_image(name_id) / 255
        sp_seg_viz = mark_boundaries(img, sp_map_np, color=(0, 1, 1))
        sp_seg_viz = (sp_seg_viz * 255).astype(np.uint8)
        cv2.imwrite(osp.join(args.output_dir, f'{name_id}.png'), sp_seg_viz[:, :, ::-1])

        # save superpixel map as csv file for evaluation protocols
        np.savetxt(osp.join(args.output_dir, f'{name_id}.csv'), sp_map_np, fmt='%d', delimiter=',')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the pretrained model')
    parser.add_argument('--data_dir', type=str, default=None, help='path to the directory of dataset')
    parser.add_argument('--output_dir', type=str, help='path to the directory for saving outputs')
    parser.add_argument('-H', '--input_image_height', type=int, default=None,
                        help='height of the image inputted to model, should be a multiple of 16')
    parser.add_argument('-W', '--input_image_width', type=int, default=None,
                        help='width of the image inputted to model, should be a multiple of 16')

    all_args = parser.parse_args()
    main(all_args)
