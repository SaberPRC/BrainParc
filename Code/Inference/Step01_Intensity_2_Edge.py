"""
TODO: Code for converting T1w image to edge maps of brain tissue
#    Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

@Create at: ShanghaiTech University
@Author: Jiameng Liu
@Contact: JiamengLiu.PRC@gmail.com
"""
import os
import ants
import argparse
import numpy as np
import SimpleITK as sitk


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


def _SoberEdge(source_img_path, target_img_path):
    # TODO: normalize data using z-score 
    origin, spacing, direction, img = _ants_img_info(source_img_path)
    img = _normalize_z_score(img)
    img = ants.from_numpy(img, origin, spacing, direction)
    ants.image_write(img, target_img_path)

    # TODO: Generate edge map through Sober Operator
    data_nii = sitk.ReadImage(target_img_path)
    origin = data_nii.GetOrigin()
    spacing = data_nii.GetSpacing()
    direction = data_nii.GetDirection()

    data_float_nii = sitk.Cast(data_nii, sitk.sitkFloat32)

    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(data_float_nii)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)

    sobel_sitk.SetOrigin(origin)
    sobel_sitk.SetSpacing(spacing)
    sobel_sitk.SetDirection(direction)

    sitk.WriteImage(sobel_sitk, target_img_path)
    return None


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting for Extract Sober Edge Map')
    parser.add_argument('--input', type=str, default='/path/to/input/T1w/image', help='Original T1 image')
    parser.add_argument('--output', type=str, default='/path/to/save/extracted/edge/map',
                        help='Persudo Extracted Brain')

    args = parser.parse_args()

    _SoberEdge(args.input, args.output)
