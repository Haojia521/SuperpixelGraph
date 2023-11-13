import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

from .u2net_modules import RSU5, RSU4, RSU4S, RSU4F


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1)
    )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class CoreModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=9):
        super(CoreModel, self).__init__()

        self.stage1 = RSU5(in_ch, 16, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU4(32, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU4S(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4F(128, 64, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 128, 256)

        self.stage4d = deconv(256, 256)
        self.stage4dc = conv(512, 256)
        self.stage3d = deconv(256, 128)
        self.stage3dc = conv(256, 128)
        self.stage2d = deconv(128, 64)
        self.stage2dc = conv(128, 64)
        self.stage1d = deconv(64, 32)
        self.stage1dc = conv(64, 32)

        self.pred_mask = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.pref_mask_sem = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, return_feature=False):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)

        hx4d = self.stage4d(hx5)
        hx4dc = self.stage4dc(torch.cat((hx4d, hx4), dim=1))

        hx3d = self.stage3d(hx4dc)
        hx3dc = self.stage3dc(torch.cat((hx3d, hx3), dim=1))

        hx2d = self.stage2d(hx3dc)
        hx2dc = self.stage2dc(torch.cat((hx2d, hx2), dim=1))

        hx1d = self.stage1d(hx2dc)
        hx1dc = self.stage1dc(torch.cat((hx1d, hx1), dim=1))

        pred = self.pred_mask(hx1dc)
        pred_sem = self.pref_mask_sem(hx1dc)

        if return_feature:
            return F.softmax(pred, dim=1), pred_sem, hx1dc
        else:
            return F.softmax(pred, dim=1), pred_sem

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]