import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import models.__init__ as init
import utils
import sys
from sklearn.metrics import roc_auc_score
import numpy as np

from tqdm import tqdm

def euclid_dist(output, target):
    
    output = output.float()
    target = target.float()

    predy = ((output / 227.0) / 227.0)
    predx = ((output % 227.0) / 227.0)

    ct = 0
    total = 0
    for j in range(100):
        ground_x = target[2*j]
        ground_y = target[2*j + 1]

        if ground_x == -1 or ground_y == -1:
            break

        temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))
        total += temp
        ct += 1

    total = total / float(ct * 1.0)

    return total

def calc_ang_err(output, target, eyes):
    total = 0

    output = output.float()
    target = target.float()

    predy = ((output / 227.0) / 227.0)
    predx = ((output % 227.0) / 227.0)
    pred_point = np.array([predx, predy])

    eye_point = eyes.numpy()

    ct = 0
    total = 0
    for j in range(100):
        ground_x = target[2*j]
        ground_y = target[2*j + 1]
        gt_point = np.array([ground_x, ground_y])

        if ground_x == -1 or ground_y == -1:
            break

        pred_dir = pred_point - eye_point
        gt_dir = gt_point - eye_point

        norm_pred = (pred_dir[0] **2 + pred_dir[1] ** 2 ) ** 0.5
        norm_gt = (gt_dir[0] **2 + gt_dir[1] ** 2 ) ** 0.5

        cos_sim = (pred_dir[0]*gt_dir[0] + pred_dir[1]*gt_dir[1]) / \
                    (norm_gt * norm_pred + 1e-6)
        cos_sim = np.maximum(np.minimum(cos_sim, 1.0), -1.0)
        ang_error = np.arccos(cos_sim) * 180 / np.pi

        total += ang_error
        ct += 1

    total = total / float(ct * 1.0)

    return total

#Loss function used is cross entropy
criterion = nn.NLLLoss().cuda()

def train(net, train_dataloader, optimizer, epoch, logger):
    net.train()
    
    running_loss = []
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        optimizer.zero_grad()

        xh, xi, xp, shifted_targets, eyes, names, eyes2, gcorrs = data

        xh = xh.cuda()
        xi = xi.cuda()
        xp = xp.cuda()
        shifted_targets = shifted_targets.cuda().squeeze()

        outputs = net(xh, xi, xp)
        total_loss = criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
        for j in range(1, len(outputs)):
            total_loss += criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])

        total_loss = total_loss / (len(outputs) * 1.0)

        total_loss.backward()
        optimizer.step()

        inputs_size = xh.size(0)
        
        running_loss.append(total_loss.item())
        if i % 100 == 99:
            logger.info('%s'%(str(np.mean(running_loss))))
            running_loss = [] 
            
class GazeOptimizer():
    
    def __init__(self, net, initial_lr, weight_decay=1e-6):
        
        self.INIT_LR = initial_lr
        self.WEIGHT_DECAY = weight_decay
        self.optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
        
    def getOptimizer(self, epoch):
        
        if epoch < 8:
            lr = self.INIT_LR
        else:
            lr = self.INIT_LR / 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = self.WEIGHT_DECAY
            
        return self.optimizer

def test(net, test_data_loader, logger):
    net.eval()
    total_error = []
    all_gt_heat = []
    all_pred_heat = []

    with torch.no_grad():
        for data in tqdm(test_data_loader, total=len(test_data_loader)):

            xh, xi, xp, targets, eyes, names, eyes2, ground_labels = data
            xh = xh.cuda()
            xi = xi.cuda()
            xp = xp.cuda()

            outputs, raw_hm = net.raw_hm(xh, xi, xp)

            pred_labels = outputs.max(1)[1] #max function returns both values and indices. so max()[0] is values, max()[1] is indices
            inputs_size = xh.size(0)

            for i in range(inputs_size):           
                distval = euclid_dist(pred_labels.data.cpu()[i], ground_labels[i])
                ang_error = calc_ang_err(pred_labels.data.cpu()[i], ground_labels[i], eyes.cpu()[i])
                total_error.append([distval, ang_error])
            #preprocess preds and gt, stack as numpy array for later auc computeation
            #batch_pred, batch_gt = utils.AUC_preprocess(raw_hm.cpu(), ground_labels, inputs_size)
            #all_pred_heat.append(batch_pred)
            #all_gt_heat.append(batch_gt)
           
    logger.info('average error: %s'%str(np.mean(np.array(total_error), axis=0)))
    return np.mean(np.array(total_error), axis=0)