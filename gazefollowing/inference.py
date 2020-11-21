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

    args = p.parse_args()

    return args

def boxes2centers(normalized_boxes):
    
    center_x = (normalized_boxes[:,0] + normalized_boxes[:,2]) / 2
    center_y = (normalized_boxes[:,1] + normalized_boxes[:,3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))
    
    return normalized_centers

def select_nearest_bbox(gazepoint, gt_bboxes, gt_labels=None):
    '''
    In: Accepts gazepoint (2,) and bboxes (n_boxes, 4), normalized from [0,1]
    Out: Returns the bbox nearest to gazepoint.
    '''
    
    centers = boxes2centers(gt_bboxes)
    
    diff = centers - gazepoint
    l2dist = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    min_idx = l2dist.argmin()
    
    if gt_labels is not None:
        predicted_class = gt_labels[min_idx]
    else:
        predicted_class = -1

    nearest_box = {
        'box' : gt_bboxes[min_idx],
        'label': predicted_class,
        'index' : min_idx
    }
    return nearest_box

def demo_random(npzfile, dataset):
    
    # Load predictions from saved npz file
    predictions = np.load(npzfile)
    gazepoints = predictions['gazepoints']

    # Get a random sample from dataset
    idx = np.random.randint(len(dataset))
    data = dataset[idx]

    # Load data from the sample
    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.cuda()), [image, face_image, gaze_field, eye_position])
    gt_position = data['gt_position']
    image_path = data['image_path']
    gt_bboxes = data['gt_bboxes']
    gt_labels = data['gt_labels']
    
    # Draw gazepoints and gt
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = eye_position
    x2, y2 = gt_position
    x3, y3 = gazepoints[idx]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x3, y3 = image_width * x3, y3 * image_height
    x1, y1, x2, y2, x3, y3 = map(int, [x1, y1, x2, y2, x3, y3])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.circle(im, (x3, y3), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 2) 
    cv2.line(im, (x1, y1), (x3, y3), [0, 165, 255], 2) 

    # Select nearest bbox given the gazepoint
    gazepoint = gazepoints[idx]
    #gt_bboxes = gt_bboxes / [image_width, image_height, image_width, image_height]
    bbox_data = select_nearest_bbox(gazepoint, gt_bboxes, gt_labels)
    nearest_bbox = bbox_data['box']

    # Scale to image size
    nearest_bbox = nearest_bbox * [image_width, image_height, image_width, image_height]
    nearest_bbox = nearest_bbox.astype(int)

    # Draw bbox of prediction
    cv2.rectangle(im, (nearest_bbox[0], nearest_bbox[1]), (nearest_bbox[2], nearest_bbox[3]), (0,165,255), 2)
    
    # Draw bbox of gt
    gaze_idx = data['gaze_idx']
    box = gt_bboxes[gaze_idx]
    nearest_bbox = box * [image_width, image_height, image_width, image_height]
    nearest_bbox = nearest_bbox.astype(int)
    cv2.rectangle(im, (nearest_bbox[0], nearest_bbox[1]), (nearest_bbox[2], nearest_bbox[3]), (255,0,0), 2)

    img = im
    save_dir = './temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = 'out_%s.png' % str(0)
    save_path = save_dir + filename
    cv2.imwrite(save_path, img)
    
    return None

def test_on_image(net, test_image_path, eye):
    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, eye)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda()), [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])


    return heatmap, f_point[0], f_point[1]

def draw_results(image_path, eyepoint, gazepoint, bboxes=None):

    # Convert to numpy arrays jic users passed lists
    eyepoint, gazepoint, gazebox = map(np.array, [eyepoint, gazepoint, bboxes])
    
    # Draw gazepoints and gt
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = eyepoint
    x2, y2 = gazepoint
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 2) 

    if bboxes is not None:
        assert(len(bboxes.shape) == 2), 'gazebox must be numpy array of shape (N,4)'
        assert(bboxes.shape[1] == 4), 'gazebox must be numpy array of shape (N,4)'

        # Select nearest bbox given the gazepoint
        bbox_data = select_nearest_bbox(gazepoint, bboxes, gt_labels=None)
        nearest_bbox = bbox_data['box']

        # Scale to image size
        nearest_bbox = nearest_bbox * [image_width, image_height, image_width, image_height]
        nearest_bbox = nearest_bbox.astype(int)

        # Draw bbox of prediction
        cv2.rectangle(im, (nearest_bbox[0], nearest_bbox[1]), (nearest_bbox[2], nearest_bbox[3]), (0,165,255), 2)

    img = im
    save_dir = './temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = 'inference_%s.png' % str(0)
    save_path = save_dir + filename
    cv2.imwrite(save_path, img)
    
    return None


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

    test_only = True
    if not test_only:
        test_and_save(net, test_data_loader)

    # Get a random sample image from the dataset
    idx =  np.random.randint(len(val_set))
    
    image_path = val_set[idx]['image_path']
    eyes = val_set[idx]['eye_position']
    bboxes = val_set[idx]['gt_bboxes']

    heatmap, x, y = test_on_image(net, image_path, eyes)
    draw_results(image_path, eyes, (x,y), bboxes)
    
if __name__ == "__main__":
    main()  