import os
import ants
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from numba import jit
import scipy.ndimage as sndi
import torch.nn.functional as F

from tqdm import tqdm
from IPython import embed
from itertools import product


mask_weight = np.zeros((160, 160, 160))
mw, mh, md = mask_weight.shape
mask_weight[(mw // 8) * 3:(mw // 8) * 5, (mh // 8) * 3:(mh // 8) * 5, (md // 8) * 3:(md // 8) * 5] = 1
mask_weight = sndi.gaussian_filter(mask_weight, sigma=mh // 8)
mask_weight = (mask_weight - mask_weight.min()) / (mask_weight.max() - mask_weight.min()) / 2 + 0.5
# mask_weight = torch.from_numpy(mask_weight)
# mask_weight = mask_weight.to('cuda')


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.max_pool = nn.AdaptiveMaxPool3d((1,1,1))

        self.fc = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//ratio, 1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channel//ratio, in_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.fc(self.avg_pool(x))
        max_pool_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_pool_out+max_pool_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernal_size=5):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernal_size, padding=kernal_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv1(x))

        return x


class CBAMBasicBlock(nn.Module):
    # TODO: basic convolutional block, conv -> batchnorm -> activate
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate=True, act='LeakyReLU'):
        super(CBAMBasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

        self.en_activate = activate

    def forward(self, x):
        output = self.bn(self.conv(x))
        output = self.ca(output) * output
        output = self.sa(output) * output

        if self.en_activate:
            return self.activate(output)
        else:
            return output


class BasicBlock(nn.Module):
    # TODO: basic convolutional block, conv -> batchnorm -> activate
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate=True, act='LeakyReLU'):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

        self.en_activate = activate

    def forward(self, x):
        if self.en_activate:
            return self.activate(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    # TODO: basic residual block established by BasicBlock
    def __init__(self, in_channels, out_channels, kernel_size, padding, nums, act='LeakyReLU'):
        '''
        TODO: initial parameters for basic residual network
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        :param kernel_size: convoluition kernel size
        :param padding: padding size
        :param nums: number of basic convolutional layer
        '''
        super(ResidualBlock, self).__init__()

        layers = list()

        for _ in range(nums):
            if _ != nums - 1:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, True, act))
            else:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, False, act))

        self.do = nn.Sequential(*layers)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.do(x)
        return self.activate(output + x)


class InputTransition(nn.Module):
    # TODO: input transition convert image to feature space
    def __init__(self, in_channels, out_channels):
        '''
        TODO: initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles
        '''
        super(InputTransition, self).__init__()
        self.trans = BasicBlock(in_channels, out_channels, 3, 1, True, 'LeakyReLU')

    def forward(self, x):
        out = self.trans(x)
        return out


class OutputTransition(nn.Module):
    # TODO: feature map convert to predict results
    def __init__(self, in_channels, out_channels, act='sigmoid'):
        '''
        TODO: initial for output transition
        :param in_channels: input feature channels
        :param out_channels: output results channels
        :param act: final activate layer sigmoid or softmax
        '''
        super(OutputTransition, self).__init__()
        assert act == 'sigmoid' or act =='softmax', \
            'final activate layer should be sigmoid or softmax, current activate is :{}'.format(act)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.act = act

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.act == 'sigmoid':
            return self.sigmoid(out)
        elif self.act == 'softmax':
            return self.softmax(out)


class DownTransition(nn.Module):
    # TODO: fundamental down-sample layer <inchannel -> 2*inchannel>
    def __init__(self, in_channels, nums, act='LeakyReLU'):
        '''
        TODO: intial for down-sample
        :param in_channels: inpuit channels
        :param nums: number of reisidual block
        '''
        super(DownTransition, self).__init__()

        out_channels = in_channels * 2
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums, act)

    def forward(self, x):
        out = self.activate1(self.bn1(self.down(x)))
        out = self.residual(out)
        return out


class UpTransition(nn.Module):
    # TODO: fundamental up-sample layer (inchannels -> inchannels/2)
    def __init__(self, in_channels, out_channels, nums):
        '''
        TODO: initial for up-sample
        :param in_channels: input channels
        :param out_channels: output channels
        :param nums: number of residual block
        '''
        super(UpTransition, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels//2)
        self.activate = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x, skip_x):
        out = self.up(x)
        out = self.activate(self.bn(self.conv1(out)))
        out = torch.cat((out,skip_x), 1)
        out = self.residual(out)

        return out


class SegNetMS_Boundary(nn.Module):
    # TODO: fundamental segmentation framework
    # Multi-Scale strategy using different crop size and normalize to same size
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_tr_s = InputTransition(in_channels, 16)
        self.in_tr_b = InputTransition(in_channels, 16)
        self.fusion_16 = CBAMBasicBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.down_32_s = DownTransition(16, 4)
        self.down_32_b = DownTransition(16, 4)
        self.fusion_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.down_64_s = DownTransition(32, 4)
        self.down_64_b = DownTransition(32, 4)
        self.fusion_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.down_128_s = DownTransition(64, 4)
        self.down_128_b = DownTransition(64, 4)
        self.fusion_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.down_256_s = DownTransition(128, 8)
        self.down_256_b = DownTransition(128, 8)
        self.fusion_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.bottleneck = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, nums=8)

        self.up_256_s = UpTransition(256, 256, 8)
        self.up_256_b = UpTransition(256, 256, 8)
        self.fusion_up_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.up_128_s = UpTransition(256, 128, 4)
        self.up_128_b = UpTransition(256, 128, 4)
        self.fusion_up_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_64_s = UpTransition(128, 64, 4)
        self.up_64_b = UpTransition(128, 64, 4)
        self.fusion_up_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.up_32_s = UpTransition(64, 32, 4)
        self.up_32_b = UpTransition(64, 32, 4)
        self.fusion_up_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.out_tr_s = OutputTransition(32, out_channels, 'sigmoid')
        self.out_tr_b = OutputTransition(32, out_channels, 'sigmoid')

    def forward(self, x):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 64, H - 64, D - 64

        x_s = x[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s], mode='trilinear')

        out_16_s = self.in_tr_s(x_s) # 96*96*96 & 128*128*128
        out_16_b = self.in_tr_b(x_b) # 96*96*96 & 128*128*128
        out_16_s = torch.cat([out_16_s, F.interpolate(out_16_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_16_s = self.fusion_16(out_16_s)

        out_32_s = self.down_32_s(out_16_s) # 48*48*48 & 64*64*64
        out_32_b = self.down_32_b(out_16_b) # 48*48*48 & 64*64*64

        out_32_s = torch.cat([out_32_s, F.interpolate(out_32_b[:, :, 10:36, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_32_s = self.fusion_32(out_32_s)

        out_64_s = self.down_64_s(out_32_s) # 24*24*24
        out_64_b = self.down_64_b(out_32_b) # 24*24*24

        out_64_s = torch.cat([out_64_s, F.interpolate(out_64_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_64_s = self.fusion_64(out_64_s)

        out_128_s = self.down_128_s(out_64_s) # 12*12*12
        out_128_b = self.down_128_b(out_64_b) # 12*12*12

        out_128_s = torch.cat([out_128_s, F.interpolate(out_128_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_128_s = self.fusion_128(out_128_s)

        out_256_s = self.down_256_s(out_128_s) # 6*6*6
        out_256_b = self.down_256_b(out_128_b) # 6*6*6

        out_256_s = torch.cat([out_256_s, F.interpolate(out_256_b[:, :, 1:5, 1:5, 1:5], size=[6, 6, 6], mode='trilinear')], dim=1)
        out_256_s = self.fusion_256(out_256_s)

        out_256_b = self.bottleneck(out_256_b)
        out_256_s = self.bottleneck(out_256_s)

        out_s = self.up_256_s(out_256_s, out_128_s)
        out_b = self.up_256_b(out_256_b, out_128_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_s = self.fusion_up_256(out_s)

        out_s = self.up_128_s(out_s, out_64_s)
        out_b = self.up_128_b(out_b, out_64_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_s = self.fusion_up_128(out_s)

        out_s = self.up_64_s(out_s, out_32_s)
        out_b = self.up_64_b(out_b, out_32_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 10:38, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_s = self.fusion_up_64(out_s)

        out_s = self.up_32_s(out_s, out_16_s)
        out_b = self.up_32_b(out_b, out_16_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_s = self.fusion_up_32(out_s)

        out_s = self.out_tr_s(out_s)
        out_b = self.out_tr_b(out_b)

        return out_s, out_b


class SegNetMS_Tissue(nn.Module):
    # TODO: fundamental segmentation framework
    # Multi-Scale strategy using different crop size and normalize to same size
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_tr_s = InputTransition(in_channels, 16)
        self.in_tr_b = InputTransition(in_channels, 16)
        self.fusion_16 = CBAMBasicBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.down_32_s = DownTransition(16, 4)
        self.down_32_b = DownTransition(16, 4)
        self.fusion_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.down_64_s = DownTransition(32, 4)
        self.down_64_b = DownTransition(32, 4)
        self.fusion_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.down_128_s = DownTransition(64, 4)
        self.down_128_b = DownTransition(64, 4)
        self.fusion_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.down_256_s = DownTransition(128, 8)
        self.down_256_b = DownTransition(128, 8)
        self.fusion_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.bottleneck = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, nums=8)

        self.up_256_s = UpTransition(256, 256, 8)
        self.up_256_b = UpTransition(256, 256, 8)
        self.fusion_up_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.up_128_s = UpTransition(256, 128, 4)
        self.up_128_b = UpTransition(256, 128, 4)
        self.fusion_up_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_64_s = UpTransition(128, 64, 4)
        self.up_64_b = UpTransition(128, 64, 4)
        self.fusion_up_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.up_32_s = UpTransition(64, 32, 4)
        self.up_32_b = UpTransition(64, 32, 4)
        self.fusion_up_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.out_tr_s = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_b = OutputTransition(32, out_channels, 'softmax')

    def forward(self, x, boundary_s, boundary_b):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 64, H - 64, D - 64

        x_s = x[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s], mode='trilinear')

        boundary_s = boundary_s[:, 1:, :, :, :]
        boundary_b = boundary_b[:, 1:, :, :, :]

        x_s = torch.cat([x_s, boundary_s], dim=1)
        x_b = torch.cat([x_b, boundary_b], dim=1)
        out_16_s = self.in_tr_s(x_s) # 96*96*96 & 128*128*128
        out_16_b = self.in_tr_b(x_b) # 96*96*96 & 128*128*128
        out_16_s = torch.cat([out_16_s, F.interpolate(out_16_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_16_s = self.fusion_16(out_16_s)

        out_32_s = self.down_32_s(out_16_s) # 48*48*48 & 64*64*64
        out_32_b = self.down_32_b(out_16_b) # 48*48*48 & 64*64*64

        out_32_s = torch.cat([out_32_s, F.interpolate(out_32_b[:, :, 10:36, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_32_s = self.fusion_32(out_32_s)

        out_64_s = self.down_64_s(out_32_s) # 24*24*24
        out_64_b = self.down_64_b(out_32_b) # 24*24*24

        out_64_s = torch.cat([out_64_s, F.interpolate(out_64_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_64_s = self.fusion_64(out_64_s)

        out_128_s = self.down_128_s(out_64_s) # 12*12*12
        out_128_b = self.down_128_b(out_64_b) # 12*12*12

        out_128_s = torch.cat([out_128_s, F.interpolate(out_128_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_128_s = self.fusion_128(out_128_s)

        out_256_s = self.down_256_s(out_128_s) # 6*6*6
        out_256_b = self.down_256_b(out_128_b) # 6*6*6

        out_256_s = torch.cat([out_256_s, F.interpolate(out_256_b[:, :, 1:5, 1:5, 1:5], size=[6, 6, 6], mode='trilinear')], dim=1)
        out_256_s = self.fusion_256(out_256_s)

        out_256_b = self.bottleneck(out_256_b)
        out_256_s = self.bottleneck(out_256_s)

        out_s = self.up_256_s(out_256_s, out_128_s)
        out_b = self.up_256_b(out_256_b, out_128_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_s = self.fusion_up_256(out_s)

        out_s = self.up_128_s(out_s, out_64_s)
        out_b = self.up_128_b(out_b, out_64_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_s = self.fusion_up_128(out_s)

        out_s = self.up_64_s(out_s, out_32_s)
        out_b = self.up_64_b(out_b, out_32_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 10:38, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_s = self.fusion_up_64(out_s)

        out_s = self.up_32_s(out_s, out_16_s)
        out_b = self.up_32_b(out_b, out_16_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_s = self.fusion_up_32(out_s)

        out_s = self.out_tr_s(out_s)
        out_b = self.out_tr_b(out_b)

        return out_s, out_b


class SegNetMS_Parc(nn.Module):
    # TODO: fundamental segmentation framework
    # Multi-Scale strategy using different crop size and normalize to same size
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_tr_s = InputTransition(in_channels, 16)
        self.in_tr_b = InputTransition(in_channels, 16)
        self.fusion_16 = CBAMBasicBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.down_32_s = DownTransition(16, 4)
        self.down_32_b = DownTransition(16, 4)
        self.fusion_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.down_64_s = DownTransition(32, 4)
        self.down_64_b = DownTransition(32, 4)
        self.fusion_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.down_128_s = DownTransition(64, 4)
        self.down_128_b = DownTransition(64, 4)
        self.fusion_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.down_256_s = DownTransition(128, 8)
        self.down_256_b = DownTransition(128, 8)
        self.fusion_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.bottleneck = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, nums=8)

        self.up_256_s = UpTransition(256, 256, 8)
        self.up_256_b = UpTransition(256, 256, 8)
        self.fusion_up_256 = CBAMBasicBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.up_128_s = UpTransition(256, 128, 4)
        self.up_128_b = UpTransition(256, 128, 4)
        self.fusion_up_128 = CBAMBasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_64_s = UpTransition(128, 64, 4)
        self.up_64_b = UpTransition(128, 64, 4)
        self.fusion_up_64 = CBAMBasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.up_32_s = UpTransition(64, 32, 4)
        self.up_32_b = UpTransition(64, 32, 4)
        self.fusion_up_32 = CBAMBasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.out_tr_s = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_b = OutputTransition(32, out_channels, 'softmax')

    def forward(self, x, tissue_s, tissue_b):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 64, H - 64, D - 64

        x_s = x[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s], mode='trilinear')

        tissue_s = tissue_s[:, 1:, :, :, :]
        tissue_b = tissue_b[:, 1:, :, :, :]

        x_s = torch.cat([x_s, tissue_s], dim=1)
        x_b = torch.cat([x_b, tissue_b], dim=1)

        out_16_s = self.in_tr_s(x_s) # 96*96*96 & 128*128*128
        out_16_b = self.in_tr_b(x_b) # 96*96*96 & 128*128*128
        out_16_s = torch.cat([out_16_s, F.interpolate(out_16_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_16_s = self.fusion_16(out_16_s)

        out_32_s = self.down_32_s(out_16_s) # 48*48*48 & 64*64*64
        out_32_b = self.down_32_b(out_16_b) # 48*48*48 & 64*64*64

        out_32_s = torch.cat([out_32_s, F.interpolate(out_32_b[:, :, 10:36, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_32_s = self.fusion_32(out_32_s)

        out_64_s = self.down_64_s(out_32_s) # 24*24*24
        out_64_b = self.down_64_b(out_32_b) # 24*24*24

        out_64_s = torch.cat([out_64_s, F.interpolate(out_64_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_64_s = self.fusion_64(out_64_s)

        out_128_s = self.down_128_s(out_64_s) # 12*12*12
        out_128_b = self.down_128_b(out_64_b) # 12*12*12

        out_128_s = torch.cat([out_128_s, F.interpolate(out_128_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_128_s = self.fusion_128(out_128_s)

        out_256_s = self.down_256_s(out_128_s) # 6*6*6
        out_256_b = self.down_256_b(out_128_b) # 6*6*6

        out_256_s = torch.cat([out_256_s, F.interpolate(out_256_b[:, :, 1:5, 1:5, 1:5], size=[6, 6, 6], mode='trilinear')], dim=1)
        out_256_s = self.fusion_256(out_256_s)

        out_256_b = self.bottleneck(out_256_b)
        out_256_s = self.bottleneck(out_256_s)

        out_s = self.up_256_s(out_256_s, out_128_s)
        out_b = self.up_256_b(out_256_b, out_128_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 3:9, 3:9, 3:9], size=[12, 12, 12], mode='trilinear')], dim=1)
        out_s = self.fusion_up_256(out_s)

        out_s = self.up_128_s(out_s, out_64_s)
        out_b = self.up_128_b(out_b, out_64_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 5:20, 5:20, 5:20], size=[24, 24, 24], mode='trilinear')], dim=1)
        out_s = self.fusion_up_128(out_s)

        out_s = self.up_64_s(out_s, out_32_s)
        out_b = self.up_64_b(out_b, out_32_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 10:38, 10:38, 10:38], size=[48, 48, 48], mode='trilinear')], dim=1)
        out_s = self.fusion_up_64(out_s)

        out_s = self.up_32_s(out_s, out_16_s)
        out_b = self.up_32_b(out_b, out_16_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 20:76, 20:76, 20:76], size=[96, 96, 96], mode='trilinear')], dim=1)
        out_s = self.fusion_up_32(out_s)

        out_s = self.out_tr_s(out_s)
        out_b = self.out_tr_b(out_b)

        return out_s, out_b


class BT_Joint(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.Boundary = SegNetMS_Boundary(2, 4)
        self.Tissue = SegNetMS_Tissue(5, 4)
        self.Parc = SegNetMS_Parc(5, 107)

    def forward(self, x):
        boundary_s, boundary_b = self.Boundary(x)
        tissue_s, tissue_b = self.Tissue(x, boundary_s, boundary_b)
        parc_s, parc_b = self.Parc(x, tissue_s, tissue_b)

        return boundary_s, boundary_b, tissue_s, tissue_b, parc_s, parc_b


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _normalize_z_score(data, clip=True):
    '''
    funtions to normalize data to standard distribution using (data - data.mean()) / data.std()
    :param data: numpy array
    :param clip: whether using upper and lower clip
    :return: normalized data by using z-score
    '''
    if clip == True:
        bounds = np.percentile(data, q=[0.00, 99.999])
        data[data <= bounds[0]] = bounds[0]
        data[data >= bounds[1]] = bounds[1]

    return (data - data.mean()) / data.std()


def calculate_patch_index(target_size, patch_size, overlap_ratio = 0.4):
    shape = target_size

    gap = int(patch_size[0] * (1-overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0]-patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1]-patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2]-patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos


def _model_init(args, model_path):
    # initialize model and load pretrained model parameters
    model = BT_Joint(args.num_modalities, args.num_classes+1)
    model = model.to(args.device)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()
    return model


@jit
def _matrix_operation(pred_rec_tissue_s, model_out_parc_s, start_pos, batch_size):
    pred_rec_tissue_s[:, start_pos[0]:start_pos[0] + batch_size[0], start_pos[1]:start_pos[1] + batch_size[1],
    start_pos[2]:start_pos[2] + batch_size[2]] += model_out_parc_s[0, :, :, :, :]
    return pred_rec_tissue_s


def _get_pred(args, model, image):

    if len(image.shape) == 4:
        img = torch.unsqueeze(image, dim=0)

    B, C, W, H, D = img.shape

    m = nn.ConstantPad3d(32, 0)
    batch_size = args.crop_size

    pos = calculate_patch_index((W, H, D), args.crop_size, overlap_ratio=args.overlap_ratio)

    pred_rec_s = np.zeros((args.num_classes+1, W, H, D))

    for start_pos in pos:
        patch = img[:,:,start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]]
        _, _, _, _, model_out_parc_s, _ = model(patch)
        model_out_parc_s = m(model_out_parc_s)
        model_out_parc_s = model_out_parc_s.cpu().detach().numpy()

        pred_rec_s = _matrix_operation(pred_rec_s, model_out_parc_s, start_pos, batch_size)

    pred_rec_s = pred_rec_s[:, 32:W-32, 32:H-32, 32:D-32]

    return pred_rec_s


def get_pred(args, model, img_path, edge_path):
    origin, spacing, direction, img = _ants_img_info(img_path)
    origin, spacing, direction, edge = _ants_img_info(edge_path)
    img = _normalize_z_score(img)
    edge = _normalize_z_score(edge)
    img = np.pad(img, ((32, 32), (32, 32), (32, 32)), 'constant')
    edge = np.pad(edge, ((32, 32), (32, 32), (32, 32)), 'constant')
    img = torch.from_numpy(img).type(torch.float32)
    edge = torch.from_numpy(edge).type(torch.float32)
    img = img.to(args.device)
    edge = edge.to(args.device)

    img = img.unsqueeze(0)
    edge = edge.unsqueeze(0)

    img = torch.cat([img, edge], dim=0)

    pred_parc = _get_pred(args, model, img)

    pred_parc = pred_parc.argmax(0)
    pred_parc = pred_parc.astype(np.float32)
    ants_img_pred_parc = ants.from_numpy(pred_parc, origin, spacing, direction)


    return ants_img_pred_parc


def _multi_layer_dice_coefficient(source, target, ep=1e-8):
    '''
    TODO: functions to calculate dice coefficient of multi class
    :param source: numpy array (Prediction)
    :param target: numpy array (Ground-Truth)
    :return: vector of dice coefficient
    '''
    class_num = (target.max()+1).astype(int)

    source = source.astype(int)
    source = np.eye(class_num.astype(int))[source]
    source = source[:,:,:,1:]
    source = source.reshape((-1, class_num-1))

    target = target.astype(int)
    target = np.eye(class_num)[target]
    target = target[:,:,:,1:]
    target = target.reshape(-1, class_num-1)

    intersection = 2 * np.sum(source * target, axis=0) + ep
    union = np.sum(source, axis=0) + np.sum(target, axis=0) + ep

    return intersection / union


def _get_pred_parallel(args, model, source, target, item):
    source_img_path = os.path.join(source, item, 'brain.nii.gz')
    source_edge_path = os.path.join(source, item, 'brain_sober.nii.gz')
    target_tissue_path = os.path.join(target, item, 'BrainParc_tissue.nii.gz')
    target_dk_path = os.path.join(target, item, 'BrainParc_dk.nii.gz')

    pred_parc = get_pred(args, model, source_img_path, source_edge_path)

    ants.image_write(pred_parc, target_dk_path)



if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Inference Setting for Brain Tissue Segmentation and region parcellation')
    parser.add_argument('--num_classes', type=int, default=106, help='number of output channels')
    parser.add_argument('--num_modalities', type=int, default=2, help='number of input channels')
    parser.add_argument('--model_path', type=str,
                        default='/public_bme/home/liujm/BrainParc/Results/Lifespan/ParcJoint/checkpoints/chk_160.pth.gz',
                        help='Pretrained model path')
    parser.add_argument('--device', type=str, default='cuda', help='specify device type: cuda or cpu?')
    parser.add_argument('--crop_size', type=tuple, default=(160, 160, 160), help='patch size')
    parser.add_argument('--overlap_ratio', type=float, default=0.6, help='Overlap ratio to extract '
                                                                          'patches for single image inference')
    parser.add_argument('--num_modalities', type=int, default=2, help='number of input channels')
    parser.add_argument('--input_brain', type=str, default='/path/to/input/image/brain/path', help='input brain data (skull-stripped)')
    parser.add_argument('--input_edge', type=str, default='/path/to/input/image/edge/path', help='input brain edge map (sobel edge)')
    parser.add_argument('--output_edge', type=str, default='/path/to/output/predicted/tissue/path', help='tissue save path')
    parser.add_argument('--output_edge', type=str, default='/path/to/output/predicted/dk/path', help='dk save path')

    args = parser.parse_args()

    model = _model_init(args, args.model_path)

    _get_pred_parallel(args, model, args.input_brain, args.input_edge, args.output_tissue, args.output_dk)

