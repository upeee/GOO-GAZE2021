import torch
from torchvision import transforms
import torch.nn as nn

from models.chong import ModelSpatial
# from dataset import GazeFollowOrig
# from config import *
# from utils import imutils, evaluation
from dataloader import chong_imutils

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
# from scipy.misc import imresize
# from tensorboardX import SummaryWriter
import warnings
from tqdm import tqdm

import cv2
from sklearn.metrics import roc_auc_score


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im,onehot_im.size), np.reshape(heatmap,heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class GazeOptimizer():

    def __init__(self, net, initial_lr, weight_decay=1e-6):

        self.INIT_LR = initial_lr
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.INIT_LR)

    def getOptimizer(self, epoch):

        if epoch < 15:
            lr = self.INIT_LR
        elif epoch < 20:
            lr = self.INIT_LR / 10
        else:
            lr = self.INIT_LR / 100

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return self.optimizer

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def train(net, train_dataloader, optimizer, epoch, logger):
    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    # step = 0
    loss_amp_factor = 10000 # multiplied to the loss to prevent underflow
    # max_steps = len(train_loader)
    optimizer.zero_grad()

    # print("Training in progress ...")
    running_loss = []

    net.train(True)
    for i, (img, face, head_channel, gaze_heatmap, name, gaze_inside) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        images = img.cuda()
        head = head_channel.cuda()
        faces = face.cuda()
        gaze_heatmap = gaze_heatmap.cuda()
        gaze_heatmap_pred, attmap, inout_pred = net(images, head, faces)
        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

        # Loss
            # l2 loss computed only for inside case
        l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor
        l2_loss = torch.mean(l2_loss, dim=1)
        l2_loss = torch.mean(l2_loss, dim=1)
        gaze_inside = gaze_inside.cuda().to(torch.float)
        l2_loss = torch.mul(l2_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
        l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside)
            # cross entropy loss for in vs out
        Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())*100

        total_loss = l2_loss #+ Xent_loss
        # NOTE: summed loss is used to train the main model.
        #       l2_loss is used to get SOTA on GazeFollow benchmark.
        total_loss.backward() # loss accumulation

        optimizer.step()
        optimizer.zero_grad()

        running_loss.append(total_loss.item())
        if i % 100 == 99:
            logger.info('%s'%(str(np.mean(running_loss, axis=0))))
            running_loss = []

    return running_loss
        # step += 1

def test(net, test_data_loader, logger, save_output=False):
    loss_amp_factor = 10000 # multiplied to the loss to prevent underflow
    net.eval()
    test_loss = []
    total_error = []

    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    all_gazepoints = []
    all_predmap = []
    all_gtmap = []

    with torch.no_grad():
        for data in tqdm(test_data_loader, total=len(test_data_loader)):
            
            val_img, val_face, val_head_channel, eye_position, val_gaze_heatmap, gaze, gaze_inside, _ = data

            val_images = val_img.cuda()
            val_head = val_head_channel.cuda()
            val_faces = val_face.cuda()
            val_gaze_heatmap = val_gaze_heatmap.cuda()
            val_gaze_heatmap_pred, val_attmap, val_inout_pred = net(val_images, val_head, val_faces)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1)

            # Loss
                # l2 loss computed only for inside case
            l2_loss = mse_loss(val_gaze_heatmap_pred, val_gaze_heatmap)*loss_amp_factor
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_inside = gaze_inside.cuda().to(torch.float)
            l2_loss = torch.mul(l2_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
            l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside)
                # cross entropy loss for in vs out
            # Xent_loss = bcelogit_loss(val_inout_pred.squeeze(), gaze_inside.squeeze())*100

            total_loss = l2_loss #+ Xent_loss
            test_loss.append(total_loss.item())

            # Obtaining eval metrics
            final_output = val_gaze_heatmap_pred.cpu().data.numpy() 
            
            gaze_x, gaze_y = gaze
            gaze_x_cpu = gaze_x.cpu().data.numpy()
            gaze_y_cpu = gaze_y.cpu().data.numpy()
            target = np.array([gaze_x_cpu, gaze_y_cpu]).T

            eye_x, eye_y = eye_position
            eye_x_cpu = eye_x.cpu().data.numpy()
            eye_y_cpu = eye_y.cpu().data.numpy()
            eye_position = np.array([eye_x_cpu, eye_y_cpu]).T

            #print('target:', target.shape)
            #print('eye_pos:', eye_position.shape)
            for f_point, gt_point, eye_point in \
                zip(final_output, target, eye_position):

                out_size = 64 #Size of heatmap for chong output
                heatmap = np.copy(f_point)
                f_point = f_point.reshape([out_size, out_size])

                h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                f_point = np.array([w_index / out_size, h_index / out_size])
                f_error = f_point - gt_point
                f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

                # angle
                f_direction = f_point - eye_point
                gt_direction = gt_point - eye_point

                norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
                norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5

                f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_f + 1e-6)
                f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
                f_angle = np.arccos(f_cos_sim) * 180 / np.pi

                #AUC calculation
                heatmap = np.squeeze(heatmap)
                heatmap = cv2.resize(heatmap, (5, 5))
                gt_heatmap = np.zeros((5, 5))
                x, y = list(map(int, gt_point * 5))
                gt_heatmap[y, x] = 1.0

                all_gazepoints.append(f_point)
                all_predmap.append(heatmap)
                all_gtmap.append(gt_heatmap)
                #score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), heatmap.reshape([-1]))

                total_error.append([f_dist, f_angle])

    l2, ang = np.mean(np.array(total_error), axis=0)

    all_gazepoints = np.vstack(all_gazepoints)
    all_predmap = np.stack(all_predmap).reshape([-1])
    all_gtmap = np.stack(all_gtmap).reshape([-1])
    auc = roc_auc_score(all_gtmap, all_predmap)

    if save_output:
        np.savez('predictions.npz', gazepoints=all_gazepoints)

    #logger.info('Test average loss : %s'%str(np.mean(np.array(test_loss), axis=0)))
    logger.info('average error: %s'%str([auc, l2, ang]))
    net.train()

    return [auc, l2, ang]


if __name__ == "__main__":
    train()
