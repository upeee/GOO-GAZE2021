import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

def F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap):
    # point loss
    heatmap_loss = nn.BCELoss()(predict_heatmap, gt_heatmap)

    # angle loss
    gt_direction = gt_position - eye_position
    middle_angle_loss = torch.mean(1 - nn.CosineSimilarity()(direction, gt_direction))

    return heatmap_loss, middle_angle_loss

class GazeOptimizer():

    def __init__(self, net, initial_lr):

        self.optimizer_s1 = optim.Adam([{'params': net.module.face_net.parameters(),
                                    'initial_lr': initial_lr},
                                    {'params': net.module.face_process.parameters(),
                                    'initial_lr': initial_lr},
                                    {'params': net.module.eye_position_transform.parameters(),
                                    'initial_lr': initial_lr},
                                    {'params': net.module.fusion.parameters(),
                                    'initial_lr': initial_lr}],
                                    lr=initial_lr, weight_decay=0.0001)
        self.optimizer_s2 = optim.Adam([{'params': net.module.fpn_net.parameters(),
                                    'initial_lr': initial_lr}],
                                    lr=initial_lr, weight_decay=0.0001)

        self.optimizer_s3 = optim.Adam([{'params': net.parameters(), 'initial_lr': initial_lr}],
                                lr=initial_lr*0.1, weight_decay=0.0001)

        self.lr_scheduler_s1 = optim.lr_scheduler.StepLR(self.optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1)
        self.lr_scheduler_s2 = optim.lr_scheduler.StepLR(self.optimizer_s2, step_size=5, gamma=0.1, last_epoch=-1)
        self.lr_scheduler_s3 = optim.lr_scheduler.StepLR(self.optimizer_s3, step_size=5, gamma=0.1, last_epoch=-1)

        self.optimizer = self.optimizer_s1

    def getOptimizer(self, epoch):

        if epoch < 7:
            lr_scheduler = self.lr_scheduler_s1
            self.optimizer = self.optimizer_s1
        elif epoch < 15:
            lr_scheduler = self.lr_scheduler_s2
            self.optimizer = self.optimizer_s2
        else:
            lr_scheduler = self.lr_scheduler_s3
            self.optimizer = self.optimizer_s3

        lr_scheduler.step()
        
        return self.optimizer

def train(net, train_dataloader, optimizer, epoch, logger):

    running_loss = []
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            map(lambda x: Variable(x.cuda()), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

        optimizer.zero_grad()

        direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

        heatmap_loss, m_angle_loss = \
            F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

        if epoch == 0:
            loss = m_angle_loss
        elif epoch >= 7 and epoch <= 14:
            loss = heatmap_loss
        else:
            loss = m_angle_loss + heatmap_loss

        loss.backward()
        optimizer.step()

        running_loss.append([heatmap_loss.item(),
                             m_angle_loss.item(), loss.item()])
        if i % 100 == 99:
            logger.info('%s'%(str(np.mean(running_loss, axis=0))))
            running_loss = []

    return running_loss
    
def test(net, test_data_loader, logger, save_output=False):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    all_gazepoints = []
    all_predmap = []
    all_gtmap = []

    with torch.no_grad():
        for data in tqdm(test_data_loader, total=len(test_data_loader)):
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                map(lambda x: Variable(x.cuda()), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

            direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])
            #curr_batch_size = predict_heatmap.shape[0]
            #predict_heatmap = torch.rand(curr_batch_size, 1, 56, 56).cuda()

            heatmap_loss, m_angle_loss = \
                F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

            loss = heatmap_loss + m_angle_loss


            total_loss.append([heatmap_loss.item(),
                            m_angle_loss.item(), loss.item()])
            #logger.info('loss: %.5lf, %.5lf, %.5lf'%( \
            #    heatmap_loss.item(), m_angle_loss.item(), loss.item()))

            middle_output = direction.cpu().data.numpy()
            final_output = predict_heatmap.cpu().data.numpy()
            target = gt_position.cpu().data.numpy()
            eye_position = eye_position.cpu().data.numpy()
            predict_heatmap = predict_heatmap.cpu().data.numpy()

            for m_direction, f_point, gt_point, eye_point, heatmap in \
                zip(middle_output, final_output, target, eye_position, predict_heatmap):
                f_point = f_point.reshape([224 // 4, 224 // 4])
                heatmaps.append(f_point)

                h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                f_point = np.array([w_index / 56., h_index / 56.])

                f_error = f_point - gt_point
                f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

                # angle
                f_direction = f_point - eye_point
                gt_direction = gt_point - eye_point

                norm_m = (m_direction[0] **2 + m_direction[1] ** 2 ) ** 0.5
                norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
                norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5

                m_cos_sim = (m_direction[0]*gt_direction[0] + m_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_m + 1e-6)
                m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
                m_angle = np.arccos(m_cos_sim) * 180 / np.pi

                f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_f + 1e-6)
                f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
                f_angle = np.arccos(f_cos_sim) * 180 / np.pi

                #AUC
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
                info_list.append(list(f_point))
                
    info_list = np.array(info_list)

    l2, ang = np.mean(np.array(total_error), axis=0)
    all_gazepoints = np.vstack(all_gazepoints)
    all_predmap = np.stack(all_predmap).reshape([-1])
    all_gtmap = np.stack(all_gtmap).reshape([-1])
    auc = roc_auc_score(all_gtmap, all_predmap)

    if save_output:
        np.savez('predictions.npz', gazepoints=all_gazepoints)

    #logger.info('average loss : %s'%str(np.mean(np.array(total_loss), axis=0))) 
    logger.info('average error: %s'%str([auc, l2, ang]))

    net.train()

    return [auc, l2, ang]