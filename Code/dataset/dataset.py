import os
import sys
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from IPython import embed
from dataset.basic import _ants_img_info, _normalize_z_score, _train_test_split, _sitk_img_info
from dataset.basic import _random_seed, _mask_seed, _crop_and_convert_to_tensor, _idx_crop


class ParcBaseMS(torch.utils.data.Dataset):
    def __init__(self, root, file_list, fold, type='train', crop_size=(96, 96, 96)):
        super().__init__()
        self.root = root
        self.type = type
        self.crop_size = crop_size
        file_list = os.path.join(root, 'csvfile', file_list)
        self.file_list = _train_test_split(file_list, fold, type)

    def __getitem__(self, idx):
        file_name, folder = self.file_list[idx][0], self.file_list[idx][1]
        img_path = os.path.join(self.root, 'data', folder, file_name, 'brain.nii.gz')
        edge_path = os.path.join(self.root, 'data', folder, file_name, 'brain_sober.nii.gz')
        tissue_path = os.path.join(self.root, 'data', folder, file_name, 'tissue.nii.gz')
        dk_path = os.path.join(self.root, 'data', folder, file_name, 'dk-struct.nii.gz')

        origin, spacing, direction, img = _ants_img_info(img_path)
        origin, spacing, direction, edge = _ants_img_info(edge_path)
        origin, spacing, direction, tissue = _ants_img_info(tissue_path)
        origin, spacing, direction, dk = _ants_img_info(dk_path)

        img = _normalize_z_score(img)
        edge = _normalize_z_score(edge)

        img = np.pad(img, ((32, 32), (32, 32), (32, 32)), 'constant')
        edge = np.pad(edge, ((32, 32), (32, 32), (32, 32)), 'constant')
        tissue = np.pad(tissue, ((32, 32), (32, 32), (32, 32)), 'constant')

        if self.type == 'train':
            dk = np.pad(dk, ((32, 32), (32, 32), (32, 32)), 'constant')

            if random.random() > 0.1:
                start_pos = _mask_seed(tissue, self.crop_size)
            else:
                start_pos = _random_seed(tissue, self.crop_size)

            img_cropped = _crop_and_convert_to_tensor(img, start_pos, self.crop_size)
            edge_cropped = _crop_and_convert_to_tensor(edge, start_pos, self.crop_size)
            tissue_cropped = _crop_and_convert_to_tensor(tissue, start_pos, self.crop_size)
            dk_cropped = _crop_and_convert_to_tensor(dk, start_pos, self.crop_size)

            tissue_cropped_one_hot = tissue_cropped.squeeze(0).type(torch.long)
            tissue_cropped_one_hot = F.one_hot(tissue_cropped_one_hot, 4)
            tissue_cropped_one_hot = tissue_cropped_one_hot.permute(3, 0, 1, 2)
            tissue_cropped_one_hot = tissue_cropped_one_hot.type(torch.float32)

            return img_cropped, edge_cropped, tissue_cropped, tissue_cropped_one_hot, dk_cropped

        elif self.type == 'val' or self.type == 'test':
            img = torch.from_numpy(img).type(torch.float32)
            edge = torch.from_numpy(edge).type(torch.float32)
            tissue = torch.from_numpy(tissue).type(torch.float32)
            dk = torch.from_numpy(dk).type(torch.float32)

            return img, edge, tissue, dk, file_name, origin, spacing, direction

    def __len__(self):
        return len(self.file_list)

