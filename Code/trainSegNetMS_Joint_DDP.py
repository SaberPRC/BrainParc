'''
Repo. for Lifespan Brain Tissue Segmentation and Region Parcellation Framework
Copy right: Jiameng Liu, ShanghaiTech University
Contact: JiamengLiu.PRC@gmail.com
'''

import os
import sys

import ants
import time
import torch
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IPython import embed
from config.config import cfg
from network.Joint_Parc_96 import BT_Joint
from torch.utils.data import DataLoader
from dataset.dataset import ParcBaseMS
from utils.loss import DiceLoss, FocalLoss, soft_cldice, soft_erode
from utils.utils import set_initial, calculate_patch_index, weights_init, logging_init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def get_pred(img, model, batch_size):
    if len(img.shape) == 4:
        img = torch.unsqueeze(img, dim=0)

    B, C, W, H, D = img.shape

    m = nn.ConstantPad3d(32, 0)

    pos = calculate_patch_index((W, H, D), batch_size, overlap_ratio=0.4)
    pred_rec_s = torch.zeros((cfg.dataset.num_classes + 1, W, H, D))

    freq_rec = torch.zeros((cfg.dataset.num_classes + 1, W, H, D))

    for start_pos in pos:
        patch = img[:, :, start_pos[0]:start_pos[0] + batch_size[0], start_pos[1]:start_pos[1] + batch_size[1],
                start_pos[2]:start_pos[2] + batch_size[2]]
        _, _, _, _, model_out_s, _ = model(patch)

        model_out_s = m(model_out_s)

        model_out_s = model_out_s.cpu().detach()

        pred_rec_s[:, start_pos[0]:start_pos[0] + batch_size[0], start_pos[1]:start_pos[1] + batch_size[1],
        start_pos[2]:start_pos[2] + batch_size[2]] += model_out_s[0, :, :, :, :]
        freq_rec[:, start_pos[0]:start_pos[0] + batch_size[0], start_pos[1]:start_pos[1] + batch_size[1],
        start_pos[2]:start_pos[2] + batch_size[2]] += 1

    pred_rec_s = pred_rec_s / freq_rec

    pred_rec_s = pred_rec_s[:, 32:W - 32, 32:H - 32, 32:D - 32]

    return pred_rec_s


def _multi_layer_dice_coefficient(source, target, ep=1e-8):
    '''
    TODO: functions to calculate dice coefficient of multi class
    :param source: numpy array (Prediction)
    :param target: numpy array (Ground-Truth)
    :return: vector of dice coefficient
    '''
    class_num = target.max() + 1

    source = source.astype(int)
    source = np.eye(class_num)[source]
    source = source[:, :, :, 1:]
    source = source.reshape((-1, class_num - 1))

    target = target.astype(int)
    target = np.eye(class_num)[target]
    target = target[:, :, :, 1:]
    target = target.reshape(-1, class_num - 1)

    intersection = 2 * np.sum(source * target, axis=0) + ep
    union = np.sum(source, axis=0) + np.sum(target, axis=0) + ep

    return intersection / union


def dice_loss(source, target, ep=1e-9):
    class_num = source.shape[1]
    source = source.reshape((-1, class_num))
    target = target.reshape((-1, class_num))

    intersection = 2 * torch.sum(source * target, axis=0) + ep
    union = torch.sum(source, axis=0) + torch.sum(target, axis=0) + ep

    return 1 - torch.mean(intersection / union)


def test(args, model, infer_data, infer_num, epoch, device=torch.device('cuda')):
    # initial model and set parameters
    model.eval()

    # setting save_path
    save_path = os.path.join(cfg.general.save_dir, 'pred', 'chk_' + str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rec = list()

    for idx in tqdm(range(infer_num // 3)):
        # Get testing data info
        img, edge, tissue, dk, img_name, origin, spacing, direction = infer_data.__getitem__(idx)

        img, edge = img.to(device), edge.to(device)
        img, edge = img.unsqueeze(0), edge.unsqueeze(0)
        x = torch.cat([img, edge], dim=0)

        pred_s = get_pred(x, model, cfg.dataset.crop_size)
        pred_s = pred_s.argmax(0)
        pred_s = pred_s.numpy().astype(np.float32)

        dk = dk.numpy().astype(int)

        dice_all = _multi_layer_dice_coefficient(pred_s, dk)

        ants_img_pred_seg = ants.from_numpy(pred_s, origin, spacing, direction)

        temp = [img_name]
        temp.extend(dice_all)
        temp.append(np.mean(np.array(dice_all)))

        rec.append(temp)

        ants.image_write(ants_img_pred_seg, os.path.join(save_path, img_name + '_seg.nii.gz'))
    df = pd.DataFrame(rec,
                      columns=['IDs', 'Precentral_L', 'Precentral_R', 'Postcentral_L', 'Postcentral_R', 'Paracentral_L',
                               'Paracentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Mid_Rostral_L',
                               'Frontal_Mid_Rostral_R', 'Frontal_Mid_Caudal_L', 'Frontal_Mid_Caudal_R', 'Frontalpole_L',
                               'Frontalpole_R', 'Orbitofrontal_Lat_L', 'Orbitofrontal_Lat_R', 'Orbitofrontal_Med_L',
                               'Orbitofrontal_Med_R', 'Parsopercularis_L', 'Parsopercularis_R', 'Parsorbitalis_L',
                               'Parsorbitalis_R', 'Parstriangularis_L', 'Parstriangularis_R', 'Insula_L', 'Insula_R',
                               'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R',
                               'Cingulum_Post_L', 'Cingulum_Post_R', 'Isthmuscingulate_L', 'Isthmuscingulate_R',
                               'Hippocampus_L', 'Hippocampus_R', 'Parahippocampal_L', 'Parahippocampal_R', 'Amygdala_L',
                               'Amygdala_R',
                               'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
                               'Thalamus_L', 'Thalamus_R', 'Accumbens_Area_L', 'Accumbens_Area_R', 'VentralDC_L',
                               'VentralDC_R', 'Choroid_Plexus_L', 'Choroid_Plexus_R', 'Ventricle_Lat_L',
                               'Ventricle_Lat_R',
                               'Ventricle_Inf_Lat_L', 'Ventricle_Inf_Lat_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                               'Parietal_Inf_L', 'Parietal_Inf_R', 'Cuneus_L', 'Cuneus_R', 'Entorhinal_L',
                               'Entorhinal_R', 'Fusiform_L', 'Fusiform_R', 'Lingual_L', 'Lingual_R', 'Pericalcarine_L',
                               'Pericalcarine_R', 'Precuneus_L', 'Precuneus_R', 'Supramarginal_L', 'Supramarginal_R',
                               'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Inf_L',
                               'Temporal_Inf_R', 'Temporalpole_L', 'Temporalpole_R', 'Temporal_Sup_Banks_L',
                               'Temporal_Sup_Banks_R', 'Transversetemporal_L', 'Transversetemporal_R',
                               'Occipital_Lat_L', 'Occipital_Lat_R', 'Cerebral_WM_L', 'Cerebral_WM_R',
                               'Cerebellum_Cortex_L', 'Cerebellum_Cortex_R', 'Cerebellum_WM_L', 'Cerebellum_WM_R',
                               'Ventricle_3rd', 'Ventricle_4th',
                               'Brainstem', 'CSF', 'Optic_Chiasm', 'CC_Anterior', 'CC_Mid_Anterior', 'CC_Central',
                               'CC_Mid_Posterior', 'CC_Posterior', 'dice_mean'])
    df.to_csv(os.path.join(save_path, str(epoch) + '.csv'), index=False)
    return None


def train(cfg, args):
    # 新增：DDP backend初始化
    # 其他获取local_rank的方法已经被弃用
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置device
    torch.cuda.set_device(local_rank)
    # 用nccl后端初始化多进程，一般都用这个
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=54000))
    # 获取device，之后的模型和张量都.to(device)
    device = torch.device("cuda", local_rank)
    # # set initial checkpoint and testing results save path
    # if cfg.general.resume_epoch == -1:
    #     set_initial(cfg)
    #
    # # set initial tensorboard save path
    # if not os.path.exists(os.path.join(cfg.general.save_dir, 'log')):
    #     os.mkdir(os.path.join(cfg.general.save_dir, 'log'))

    dk_2_tissue = pd.read_csv('/public/home/liujm1/perl5/uii_dk_2_tissue.csv', encoding='windows-1252')
    gm = np.where(np.array(list(dk_2_tissue['UII_Tissue'])) == 2)[0]
    wm = np.where(np.array(list(dk_2_tissue['UII_Tissue'])) == 3)[0]
    csf = np.where(np.array(list(dk_2_tissue['UII_Tissue'])) == 1)[0]

    gm = torch.from_numpy(gm).type(torch.LongTensor).to('cuda')
    wm = torch.from_numpy(wm).type(torch.LongTensor).to('cuda')
    csf = torch.from_numpy(csf).type(torch.LongTensor).to('cuda')

    logging_init(args.log_name, PARENT_DIR=os.path.join(cfg.general.save_dir, 'log'))
    logging.info('Start training on {} server'.format(args.platform))
    logging.info('file list for training and validation: {}'.format(cfg.general.file_list))
    logging.info('file path to save all images: {}'.format(cfg.general.save_dir))
    logging.info('project level path: {}'.format(cfg.general.root))

    # Default tensor type
    torch.set_default_dtype(torch.float32)

    # Set numpy and torch seeds
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(cfg.general.seed)

    training_set = ParcMSNPY(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold,
                             type='train')
    training_sample = DistributedSampler(training_set, shuffle=True)
    training_loader = DataLoader(training_set, sampler=training_sample, batch_size=cfg.train.batch_size, num_workers=4,
                                 pin_memory=True, drop_last=True)

    infer_data = ParcMSNPY(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold,
                           type='val')
    infer_num = infer_data.__len__()

    # Init resume_spoch == -1, train from scratch
    if cfg.general.resume_epoch == -1:
        model = BT_Joint(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        weights_init(model)
        model = model.to(device)
        model = DDP(model, broadcast_buffers=False, find_unused_parameters=False)
    else:
        model = BT_Joint(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        model_path = os.path.join(cfg.general.save_dir, 'checkpoints',
                                  'chk_' + str(cfg.general.resume_epoch) + '.pth.gz')
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(local_rank)))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_focal_boundary = torch.nn.BCELoss()
    loss_dice_boundary = soft_cldice()

    loss_focal_tissue = FocalLoss(4)
    loss_dice_tissue = DiceLoss(class_num=4)

    loss_bce = torch.nn.BCELoss()

    # Loss function for optimize
    if args.weight == 1:
        loss_focal_parc = FocalLoss(cfg.dataset.num_classes + 1, alpha=cfg.train.alpha)
        loss_dice_parc = DiceLoss(alpha=cfg.train.alpha, class_num=107)
    else:
        loss_focal_parc = FocalLoss(cfg.dataset.num_classes + 1)
        loss_dice_parc = DiceLoss()

    # Optimization strategy
    if cfg.train.batch_size <= 8:
        lr = 1e-3
    else:
        lr = 2e-3

    for epoch in range(cfg.general.resume_epoch + 1, cfg.train.num_epochs):
        # training process
        if epoch <= 20:
            lr = lr
        elif epoch > 20 and epoch <= 200:
            lr = 5e-4
        elif epoch > 200:
            lr = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        model.train()
        print('Training epoch {}/{}'.format(epoch, cfg.train.num_epochs))
        for idx, (img, edge, tissue, tissue_one_hot, dk) in enumerate(training_loader):
            try:
                # start train one iteration
                x = torch.cat([img, edge], dim=1)

                x, y = x.to(device), dk.to(device)
                B, C, W, H, D = y.shape

                # TODO: define ground-truth of tissue
                tissue_s = tissue[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
                tissue_b = F.interpolate(tissue, size=[W - 64, H - 64, D - 64])
                tissue_s = tissue_s.squeeze(1)
                tissue_b = tissue_b.squeeze(1)
                # TODO: define ground-truth of region
                parc_s = y[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
                parc_b = F.interpolate(y, size=[W - 64, H - 64, D - 64])
                parc_s = parc_s.squeeze(1)
                parc_b = parc_b.squeeze(1)
                # TODO: define tissue boundary
                tissue_one_hot_s = tissue_one_hot[:, :, 32:W - 32, 32:H - 32, 32:D - 32]
                tissue_one_hot_s = tissue_one_hot_s.to(device)
                tissue_one_hot_s_edge = tissue_one_hot_s - soft_erode(tissue_one_hot_s)
                tissue_one_hot_b = F.interpolate(tissue_one_hot, size=[W - 64, H - 64, D - 64])
                tissue_one_hot_b = tissue_one_hot_b.to(device)
                tissue_one_hot_b_edge = tissue_one_hot_b - soft_erode(tissue_one_hot_b)

                optimizer.zero_grad()

                boundary_s_pred, boundary_b_pred, tissue_s_pred, tissue_b_pred, parc_s_pred, parc_b_pred = model(x)

                tissue_s_pred_boundary = tissue_s_pred - soft_erode(tissue_s_pred)
                tissue_b_pred_boundary = tissue_s_pred - soft_erode(tissue_b_pred)

                parc_s_bg, parc_b_bg = parc_s_pred[:,0,:,:,:].unsqueeze(1), parc_b_pred[:,0,:,:,:].unsqueeze(1)

                parc_s_gm = torch.index_select(parc_s_pred[:, 1:, :, :, :], dim=1, index=gm).sum(dim=1).unsqueeze(1)
                parc_b_gm = torch.index_select(parc_b_pred[:, 1:, :, :, :], dim=1, index=gm).sum(dim=1).unsqueeze(1)

                parc_s_wm = torch.index_select(parc_s_pred[:, 1:, :, :, :], dim=1, index=wm).sum(dim=1).unsqueeze(1)
                parc_b_wm = torch.index_select(parc_b_pred[:, 1:, :, :, :], dim=1, index=wm).sum(dim=1).unsqueeze(1)

                parc_s_csf = torch.index_select(parc_s_pred[:, 1:, :, :, :], dim=1, index=csf).sum(dim=1).unsqueeze(1)
                parc_b_csf = torch.index_select(parc_b_pred[:, 1:, :, :, :], dim=1, index=csf).sum(dim=1).unsqueeze(1)

                parc_s_tissue = torch.cat([parc_s_bg, parc_s_csf, parc_s_gm, parc_s_wm], dim=1)
                parc_b_tissue = torch.cat([parc_b_bg, parc_b_csf, parc_b_gm, parc_b_wm], dim=1)

                loss_similarity = (torch.abs((tissue_s_pred_boundary-boundary_s_pred)).mean() + torch.abs((tissue_b_pred_boundary-boundary_b_pred)).mean() + torch.abs((parc_s_tissue-tissue_s_pred)).mean() + torch.abs((parc_b_tissue-tissue_b_pred)).mean())/4

                loss_tissue_dice_s = loss_dice_tissue(tissue_s_pred, tissue_s)
                loss_tissue_dice_b = loss_dice_tissue(tissue_b_pred, tissue_b)
                loss_tissue_focal_s = loss_focal_tissue(tissue_s_pred, tissue_s)
                loss_tissue_focal_b = loss_focal_tissue(tissue_b_pred, tissue_b)
                loss_tissue_all = 0.7*(loss_tissue_dice_s+loss_tissue_focal_s*10) + 0.3*(loss_tissue_dice_b+loss_tissue_focal_b*10)

                loss_boundary_dice_s = loss_dice_boundary(boundary_s_pred, tissue_one_hot_s_edge)
                loss_boundary_dice_b = loss_dice_boundary(boundary_b_pred, tissue_one_hot_b_edge)
                loss_boundary_focal_s = loss_focal_boundary(boundary_s_pred, tissue_one_hot_s_edge)
                loss_boundary_focal_b = loss_focal_boundary(boundary_b_pred, tissue_one_hot_b_edge)
                loss_boundary_all = 0.7*(loss_boundary_dice_s+loss_boundary_focal_s*10) + 0.3*(loss_boundary_dice_b+loss_boundary_focal_b*10)

                loss_parc_dice_s = loss_dice_parc(parc_s_pred, parc_s)
                loss_parc_dice_b = loss_dice_parc(parc_b_pred, parc_b)
                loss_parc_focal_s = loss_focal_parc(parc_s_pred, parc_s)
                loss_parc_focal_b = loss_focal_parc(parc_b_pred, parc_b)
                loss_parc_all = 0.7*(loss_parc_dice_s+loss_parc_focal_s*10) + 0.3*(loss_parc_dice_b+loss_parc_focal_b*10)

                loss = loss_boundary_all*0.3 + loss_tissue_all*0.3 + loss_parc_all*0.4 + loss_similarity
                msg = 'epoch: {}, batch: {}, learning_rate: {}, loss: {:.4f}, loss_boundary: {:.4f}, loss_tissue: {:.4f}, loss_parc: {:.4f}' \
                    .format(epoch, idx, optimizer.param_groups[0]['lr'], loss.item(), loss_boundary_all.item(),
                            loss_tissue_all.item(), loss_parc_all.item())

                loss.backward()
                optimizer.step()
                logging.info(msg)

                if epoch != 0 and epoch % cfg.train.save_epoch == 0:
                    save_path = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '.pth.gz')
                    if idx == 0 and epoch >= 0:
                        torch.save(model.module.state_dict(), save_path)
                        # torch.save(model.state_dict(), save_path)
                        # test(args, model, infer_data, infer_num, epoch)
            except:
                print(idx)

        print('Done!')
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infant brain tissue segmentation Experiment settings')
    parser.add_argument('--platform', type=str, default='bme', help='specify compute platform')
    parser.add_argument('--fold', type=bool, default=1, help='specify testing fold')
    parser.add_argument('--save_path', type=str, default='JointParc', help='specify results save folder')
    parser.add_argument('--file_list', type=str, default='file_list_sp.csv', help='specify training file list')
    parser.add_argument('--resume', type=int, default=-1, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=2, help='number of input channels')
    parser.add_argument('--log_name', type=str, default='JointParcV2', help='number of input channels')
    parser.add_argument('--weight', type=int, default=1, help='number of input channels')

    args = parser.parse_args()

    if args.platform == 'bme':
        cfg.general.file_list = args.file_list
        cfg.general.save_dir = os.path.join(cfg.general_bme.save_root, args.save_path)
        cfg.general.root = cfg.general_bme.root

    elif args.platform == 'local':
        cfg.general.file_list = args.file_list
        cfg.general.save_dir = os.path.join(cfg.general_local.save_root, args.save_path)
        cfg.general.root = cfg.general_local.root

    cfg.dataset.crop_size = [160, 160, 160]
    cfg.general.resume_epoch = args.resume
    cfg.train.batch_size = args.batch_size

    train(cfg, args)