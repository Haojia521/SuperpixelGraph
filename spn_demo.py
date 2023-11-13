import cv2
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries

from datasets.transforms import ToTensor
from spn_test import get_input_size
from spn_train import SuperPixelNet
from utils import get_superpixel_map


def main(args):
    # prepare image data
    name_id = osp.splitext(osp.basename(args.image_path))[0]

    origin_image = cv2.cvtColor(cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(ToTensor()(origin_image))

    image_mean = torch.mean(torch.flatten(image, 1), dim=1).reshape(-1, 1, 1)
    image = image - image_mean
    image = image.cuda().unsqueeze(0)
    ori_h, ori_w = image.shape[-2:]

    input_h = get_input_size(args.input_image_height, ori_h)
    input_w = get_input_size(args.input_image_width, ori_w)

    if ori_h != input_h or ori_w != input_w:
        image = F.interpolate(image, [input_h, input_w], mode='bicubic', align_corners=True)

    # load model
    spn_model = SuperPixelNet.load_from_checkpoint(args.model_path, map_location='cpu')
    spn_model = spn_model.eval().cuda()

    downsize = spn_model.hparams.downsize

    # predict
    with torch.no_grad():
        pred_q, _ = spn_model(image)

    superpixel_map = get_superpixel_map(input_h, input_w, downsize, 1, pred_q, pred_q.device)

    if ori_h != input_h or ori_w != input_w:
        superpixel_map = F.interpolate(superpixel_map.float(), [ori_h, ori_w], mode='nearest').long()

    sp_map_np = superpixel_map.squeeze().cpu().numpy()

    # save superpixel map
    os.makedirs(args.output_dir, exist_ok=True)

    img = origin_image / 255
    sp_seg_viz = mark_boundaries(img, sp_map_np, color=(0, 1, 1))
    sp_seg_viz = (sp_seg_viz * 255).astype(np.uint8)
    cv2.imwrite(osp.join(args.output_dir, f'{name_id}.png'), sp_seg_viz[:, :, ::-1])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the pretrained model')
    parser.add_argument('--image_path', type=str, default=None, help='path to image')
    parser.add_argument('--output_dir', type=str, help='path to the directory for saving outputs')
    parser.add_argument('-H', '--input_image_height', type=int, default=None,
                        help='height of the image inputted to model, should be a multiple of 16')
    parser.add_argument('-W', '--input_image_width', type=int, default=None,
                        help='width of the image inputted to model, should be a multiple of 16')

    all_args = parser.parse_args()
    main(all_args)
