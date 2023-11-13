import os
import os.path as osp

import torch
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as ptl
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import get_dataset_trainval
from datasets.transforms import RandomFlip, Transpose, Rotate, RandomCrop, CenterCrop, ToTensor, ComposeTransforms
from model import CoreModel
from utils import generate_target_tensor, get_superpixel_map, compute_data_for_superpixel_boundary_recall
from utils import compute_loss_sp, compute_loss_se


class SuperPixelNet(ptl.LightningModule):
    def __init__(self, cfg):
        super(SuperPixelNet, self).__init__()

        self.save_hyperparameters(cfg)
        self.model = CoreModel()

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = generate_target_tensor(y)

        pred_q, pred_sem = self.model(x)
        loss_sp, loss_sp_sem, loss_sp_pos = compute_loss_sp(pred_q, target,
                                                            self.hparams.downsize, self.hparams.pos_weight)

        self.log('train_loss_sp', loss_sp, on_epoch=True)
        self.log('train_loss_sp_sem', loss_sp_sem, on_epoch=True)
        self.log('train_loss_sp_pos', loss_sp_pos, on_epoch=True)

        if self.hparams.use_loss_se:
            loss_se = compute_loss_se(pred_sem, y)
            loss_se = loss_se * self.hparams.sem_weight
            self.log('train_loss_se', loss_se, on_epoch=True)

            return loss_sp + loss_se

        return loss_sp

    def validation_step(self, batch, batch_idx):
        x, y = batch
        target = generate_target_tensor(y)

        pred_q, pred_sem = self.model(x)
        loss_sp, loss_sp_sem, loss_sp_pos = compute_loss_sp(pred_q, target,
                                                            self.hparams.downsize, self.hparams.pos_weight)

        self.log('val_loss_sp', loss_sp)
        self.log('val_loss_sp_sem', loss_sp_sem)
        self.log('val_loss_sp_pos', loss_sp_pos)

        loss = loss_sp
        if self.hparams.use_loss_se:
            loss_se = compute_loss_se(pred_sem, y)
            loss_se = loss_se * self.hparams.sem_weight
            self.log('val_loss_se', loss_se)

            loss = loss + loss_se

        b, _, h, w = x.shape
        sp_map = get_superpixel_map(h, w, self.hparams.downsize, b, pred_q, device=x.device)

        br_tp = br_fn = 0
        for nb in range(b):
            curr_sp_map_np = sp_map[nb].squeeze().cpu().numpy()
            curr_gt_np = y[nb].cpu().int().numpy()

            tp, fn = compute_data_for_superpixel_boundary_recall(curr_sp_map_np, curr_gt_np)

            br_tp += tp
            br_fn += fn

        return {'loss': loss, 'br_tp': br_tp, 'br_fn': br_fn}

    def validation_epoch_end(self, outputs):
        br_tp = sum([d['br_tp'] for d in outputs])
        br_fn = sum([d['br_fn'] for d in outputs])

        br = br_tp / (br_tp + br_fn + 1e-8)

        self.log('val_br', br)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer


def main(args):
    # crate model
    if args.pretrained_model is not None and args.load_weights_only:
        model_data = torch.load(args.pretrained_model, map_location='cpu')
        model = SuperPixelNet(args)
        model.load_state_dict(model_data['state_dict'])
        args.pretrained_model = None
    else:
        model = SuperPixelNet(args)

    # prepare datasets and data loaders
    # - data transform
    transform_train = ComposeTransforms([
        RandomCrop(height=args.train_image_h, width=args.train_image_w),
        RandomFlip(),
        Transpose(),
        Rotate(),
        ToTensor(),
    ])

    transform_val = ComposeTransforms([
        CenterCrop(height=args.val_image_h, width=args.val_image_w),
        ToTensor(),
    ])

    # - create datasets for training and validation
    ds_train, ds_val = get_dataset_trainval(args.dataset, args.data_dir, transform_train, transform_val)

    dl_train = data.DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.workers,
                               shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True)
    dl_val = data.DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                             persistent_workers=True)

    # create the callback that saves checkpoint model
    ckp_callback_val_br = ModelCheckpoint(filename='{epoch}-{val_br:.6f}',
                                          monitor='val_br', mode='max', save_top_k=3)
    ckp_callback_val_loss = ModelCheckpoint(filename='{epoch}-{val_loss_sp_sem:.6f}',
                                            monitor='val_loss_sp_sem', mode='min')

    # crate root directory for saving files
    root_dir = osp.join(args.output_dir, args.dataset)
    os.makedirs(root_dir, exist_ok=True)

    # calculate log frequency
    one_epoch_steps_min = min(len(ds_train) // args.batch_size, len(ds_val) // args.batch_size)
    log_in_steps = one_epoch_steps_min if one_epoch_steps_min < 50 else 50

    # gpus index list used for training
    gpu_ids = eval(f'[{args.gpus}]')

    # start training
    trainer = ptl.Trainer(gpus=gpu_ids,
                          accelerator='gpu',
                          num_sanity_val_steps=2 if args.check_val_sanity else 0,
                          default_root_dir=root_dir,
                          callbacks=[ckp_callback_val_br, ckp_callback_val_loss],
                          max_epochs=args.num_epochs,
                          max_steps=args.num_steps,
                          log_every_n_steps=log_in_steps,
                          limit_train_batches=args.limit_train_batches,
                          limit_val_batches=args.limit_val_batches)
    trainer.fit(model, dl_train, dl_val, ckpt_path=args.pretrained_model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='the directory of dataset')
    parser.add_argument('--output_dir', type=str, help='the directory to save outputs')
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--pos_weight', type=float, default=0.003, help='weight of pos term in loss_sp')
    parser.add_argument('--sem_weight', type=float, default=1.0, help='weight of loss_se')
    parser.add_argument('--gpus', type=str, default='0', help='gpu id(s) separated by commas. e.g. 1,3')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--downsize', type=int, default=16, help='grid cell size for initial superpixel')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train_image_h', type=int, default=320, help='image height for training')
    parser.add_argument('--train_image_w', type=int, default=320, help='image width for training')
    parser.add_argument('--val_image_h', type=int, default=480, help='image height for validation')
    parser.add_argument('--val_image_w', type=int, default=480, help='image width for validation')
    parser.add_argument('--check_val_sanity', action='store_true',
                        help='sanity check before starting the training routine')
    parser.add_argument('--use_loss_se', action='store_true', help='use loss_se')
    parser.add_argument('--num_epochs', type=int, default=None, help='the number of epochs for training')
    parser.add_argument('--num_steps', type=int, default=-1, help='the number of steps for training')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='the proportion of the number of batches used for training')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='the proportion of the number of batches used for validation')
    parser.add_argument('--pretrained_model', type=str, default=None, help='path of pretrained model')
    parser.add_argument('--load_weights_only', action='store_true',
                        help='load parameters from pretrained model rather than resuming the training')

    all_args = parser.parse_args()

    main(all_args)
