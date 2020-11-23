import sys
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
from scipy import signal
from utils import data_transforms
from tqdm import tqdm

from utils import preprocess_image
import argparse
from inference import select_nearest_bbox
from sklearn.metrics import roc_auc_score

from models.gazenet import GazeNet
from models.__init__ import save_checkpoint, resume_checkpoint
from dataloader.gazenet import GooDataset, GazeDataset

def parse_inputs():

    # Command line arguments
    p = argparse.ArgumentParser()

    p.add_argument('--test_dir', type=str,
                        help='path to test set',
                        default=None)
    p.add_argument('--test_annotation', type=str,
                        help='test annotations (pickle/mat)',
                        default=None)
    p.add_argument('--resume_path', type=str,
                        help='load model file',
                        default=None)
    p.add_argument('--predictions_npz', type=str,
                        help='.npz filepath containing predictions of the model',
                        default=None)             

    args = p.parse_args()

    return args

def test_and_save(net, test_data_loader, outfile='predictions.npz'):
    net.eval()

    all_heatmaps = []
    all_gazepoints = []
    with torch.no_grad():
        for data in tqdm(test_data_loader, total=len(test_data_loader)):
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                map(lambda x: Variable(x.cuda()), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

            direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

            final_output = predict_heatmap.cpu().data.numpy()
            target = gt_position.cpu().data.numpy()
            all_heatmaps.append(final_output)

            for f_point, gt_point, eye_point in \
                zip(final_output, target, eye_position):
                f_point = f_point.reshape([224 // 4, 224 // 4])

                h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                f_point = np.array([w_index / 56., h_index / 56.])             
                all_gazepoints.append(f_point)

                f_error = f_point - gt_point
                f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

    all_heatmaps = np.vstack(all_heatmaps).squeeze()
    all_gazepoints = np.vstack(all_gazepoints)
    np.savez(outfile, heatmaps=all_heatmaps, gazepoints=all_gazepoints)
    
    return

def calculate_metrics(npzfile, dataset):
    predictions = np.load(npzfile)
    
    error = []
    percent_dists=[0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    PA_count = np.zeros((len(percent_dists)))
    CPA_count = np.zeros((len(percent_dists)))
    AUC_score = []
    angular_err = []
    
    heatmaps = predictions['heatmaps']
    gazepoints = predictions['gazepoints']
    
    for idx, f_point in tqdm(enumerate(gazepoints), total=len(dataset)):
        
        data = dataset[idx]
        
        # Calculate L2, and use for pa/cpa
        gt_point = data['gt_position'].numpy()
        f_error = f_point - gt_point
        f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)
        error.append(f_dist)
        
        #Calc pa
        PA_count[np.array(percent_dists) > f_dist] += 1   
        
        #Calc cpa
        gt_bboxes, gt_labels, gaze_class = data['gt_bboxes'], data['gt_labels'], data['gaze_class']
        selected_bbox = select_nearest_bbox(f_point, gt_bboxes, gt_labels)
        if selected_bbox['label'] == gaze_class: 
            CPA_count[np.array(percent_dists) > f_dist] += 1   
            
        #Calc AUC
        heatmap_pred = heatmaps[idx, :, :]
        heatmap_pred = cv2.resize(heatmap_pred, (5, 5))
        gt_heatmap = np.zeros((5, 5))
        x, y = list(map(int, gt_point * 5))
        gt_heatmap[y, x] = 1.0
        
        score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), heatmap_pred.reshape([-1]))
        AUC_score.append(score)
        
        #Calc Angular error
        eye_point = data['eye_position'].numpy()
        f_direction = f_point - eye_point
        gt_direction = gt_point - eye_point
        norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
        norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5
        
        f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_f + 1e-6)
        f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
        f_angle = np.arccos(f_cos_sim) * 180 / np.pi
        angular_err.append(f_angle)

        
    l2 = np.mean(np.array(error))
    pa = PA_count / len(dataset)
    cpa = CPA_count / len(dataset)
    auc = np.mean(AUC_score)
    ang_err = np.mean(angular_err)
    metrics = {
        'auc' : auc,
        'l2' : l2,
        'angular' : ang_err,
        'pa' : pa,
        'cpa': cpa,
    }
    
    return metrics

def main():

    args = parse_inputs()

    # Load Model
    net = GazeNet()
    net.cuda()

    resume_path = args.resume_path
    net, optimizer, start_epoch = resume_checkpoint(net, None, resume_path)

    #Prepare dataloaders
    test_images_dir = args.test_dir
    test_pickle_path = args.test_annotation

    #For GOO
    val_set = GooDataset(test_images_dir, test_pickle_path, 'test', use_bboxes=True)
    test_set = GooDataset(test_images_dir, test_pickle_path, 'test')
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=16, num_workers=8, shuffle=False)

    predictions_npz = args.predictions_npz
    if predictions_npz is None:
        print('==> No npzfile provided. Inference will be done on the test dataset and will be saved to predictions.npz')
        test_and_save(net, test_data_loader)
        predictions_npz = './predictions.npz'
    
    print('==> Calculating eval metrics...')
    metrics = calculate_metrics(predictions_npz, val_set)

    PA_count = metrics['pa']
    CPA_count = metrics['cpa']
    print('AUC:', metrics['auc'])
    print('L2 Distance: ', metrics['l2'])
    print('Angular Error:', metrics['angular'])
    print("Percentage Distances: ", [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    print("Proximate Accuracy: \t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t"%
                (PA_count[0],
                PA_count[1],
                PA_count[2],
                PA_count[3],
                PA_count[4],
                PA_count[5],
                PA_count[6],
                PA_count[7],
                ))
    print("Class Proximate Accuracy: \t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t"%
                (CPA_count[0],
                CPA_count[1],
                CPA_count[2],
                CPA_count[3],
                CPA_count[4],
                CPA_count[5],
                CPA_count[6],
                CPA_count[7],
                ))
    
if __name__ == "__main__":
    main()  