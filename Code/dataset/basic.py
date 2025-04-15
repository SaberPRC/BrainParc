import os
import sys

import ants
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
from skimage import measure
from itertools import product
from scipy.ndimage import zoom


def _ants_img_info(img_path):
    '''Function to get medical image information'''
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _seg_to_label(seg):
    '''
    Labeling each single annotation in one image (single label to multiple label)
    :param seg: single label annotation (numpy data)
    :return: multiple label annotation
    '''
    labels, num = measure.label(seg, return_num=True)
    return labels, num


def _random_seed(mask, patch_size):
    '''
    TODO: Functions to get random crop seed coordinate
    :param mask: corresponding mask (array data)
    :param patch_size: crop size of extracted patch
    '''
    # get maximum index and random get selected data
    max_pos = mask.shape

    # select center point
    select_pos = [random.choice(range(max_pos[0])), random.choice(range(max_pos[1])),
                  random.choice(range(max_pos[2]))]

    gap = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
    start_pos = [select_pos[0] - gap[0], select_pos[1] - gap[1], select_pos[2] - gap[2]]
    # determine whether start position out of range
    start_pos = [max(0, f) for f in start_pos]
    # determine whether end position out of range
    for idx in range(len(start_pos)):
        if start_pos[idx] + patch_size[idx] > max_pos[idx]:
            start_pos[idx] = max_pos[idx] - patch_size[idx]

    return start_pos


def _mask_seed(mask, patch_size):
    '''
    Functions to extract seed around mask
    :param mask: corresponding mask (array data)
    :param patch_size: crop size of extracted patch
    :return: cropped image by mask and exact size
    '''
    max_pos = mask.shape

    # get potential exists center point within mask area
    index = np.where(mask != 0)
    index = list(map(list, zip(*index)))

    # select center point randomly from mask region
    select_pos = random.choice(index)
    gap = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]

    # create start position according to the center point and patch size and determine whether the start position out of range
    start_pos = [select_pos[0] - gap[0], select_pos[1] - gap[1], select_pos[2] - gap[2]]
    start_pos = [max(0, f) for f in start_pos]

    # create end position according to start point and patch size, adn determine whether the end point out of range
    for idx in range(len(start_pos)):
        if start_pos[idx] + patch_size[idx] > max_pos[idx]:
            start_pos[idx] = max_pos[idx] - patch_size[idx]
    return start_pos


def _idx_crop(img, start_pos, crop_size):
    img = img[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1],
          start_pos[2]:start_pos[2] + crop_size[2]]
    return img


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


def _sitk_img_info(img_path):
    '''
    TODO: Functions to load medical image using SimpleITK and get fundamental informations
    :param img_path: absolute image path end with: .nii.gz, .mhd, .hdr
    :return: origin, spacing, direction, image numpy array
    '''
    img = sitk.ReadImage(img_path)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    data = sitk.GetArrayFromImage(img)
    data = data.transpose(2,1,0)
    return origin, spacing, direction, data



def _train_test_split(file_list, fold, type='train'):
    test_fold = fold
    if fold + 1 > 5:
        val_fold = 1
    else:
        val_fold = fold + 1
    file_list = pd.read_csv(file_list)
    if type == 'train':
        print('train phase with test fold: {}, validation fold: {}'.format(test_fold, val_fold))
        file_list = file_list.loc[file_list['fold'] != val_fold]
        file_list = file_list.loc[file_list['fold'] != test_fold]
        file_list = file_list.values.tolist()
    elif type == 'val':
        print('validation phase with fold: {}'.format(val_fold))
        file_list = file_list.loc[file_list['fold'] == val_fold]
        file_list = file_list.values.tolist()
    elif type == 'test':
        print('testing phase with fold: {}'.format(test_fold))
        file_list = file_list.loc[file_list['fold'] == test_fold]
        file_list = file_list.values.tolist()
    return file_list


def _crop_and_convert_to_tensor(img, start_pos, crop_size):
    cropped_img = _idx_crop(img, start_pos, crop_size)
    cropped_img = cropped_img[np.newaxis, ...]
    return torch.from_numpy(cropped_img).type(torch.float32)