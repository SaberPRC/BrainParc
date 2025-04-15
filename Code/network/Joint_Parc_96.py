import os
import sys

dir_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_test)

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.basic import ResidualBlock, InputTransition, OutputTransition, DownTransition, UpTransition, BasicBlock, CBAMBasicBlock


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