import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from models.gazenet import GazeNet

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
from scipy.io import loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map
import pickle
from skimage import io


class GooDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train', include_path=False, use_gazemask=False, alpha=1.0):
        assert (training in set(['train', 'test']))
        print("Initializing GOO Dataset...")
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training
        self.include_path = include_path
        self.use_gazemask = use_gazemask
        self.alpha = alpha      # setting alpha = 1 defaults to using fmax instead of alpha blending

        with open(mat_file, 'rb') as f:
            self.data = pickle.load(f)
            self.image_num = len(self.data)

        print("Number of Images:", self.image_num)
        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):

        data = self.data[idx]
        # print("keys",data.keys())
        image_path = data['filename']
        image_path = os.path.join(self.root_dir, image_path)
        # print(image_path)

        eye = [float(data['hx']) / 640, float(data['hy']) / 480]
        gaze = [float(data['gaze_cx']) / 640, float(data['gaze_cy']) / 480]
        # print('eye coords: ', eye)
        # print('gaze coords: ', gaze)

        image_path = image_path.replace('\\', '/')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        flip_flag = False
        if random.random() > 0.5 and self.training == 'train':
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)
            flip_flag = True
            # print("FLIPPED!")

        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
        h, w = image.shape[:2]
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)
        # process image for saliency net
        # image = image_preprocess(image)
        original_image = np.copy(image)
        original_image = cv2.resize(original_image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))

        # Get additional gaze data
        gaze_idx = np.copy(data['gazeIdx'])
        gaze_class = np.copy(data['gaze_item']).astype(np.int64)

        gaze_bbox = np.copy(data['ann']['bboxes'][gaze_idx])

        # normalize
        gaze_bbox[0] = gaze_bbox[0] / 640.0
        gaze_bbox[1] = gaze_bbox[1] / 480.0
        gaze_bbox[2] = gaze_bbox[2] / 640.0
        gaze_bbox[3] = gaze_bbox[3] / 480.0

        head_bbox = np.copy(data['ann']['bboxes'][-1])   # last in the list is the head bbox

        # normalize
        head_bbox[0] = head_bbox[0] / 640.0
        head_bbox[1] = head_bbox[1] / 480.0
        head_bbox[2] = head_bbox[2] / 640.0
        head_bbox[3] = head_bbox[3] / 480.0

        # flip bbox locations
        if flip_flag:
            # flip gaze bbox
            xmin = gaze_bbox[0]
            xmax = gaze_bbox[2]
            gaze_bbox[0] = 1 - xmax
            gaze_bbox[2] = 1 - xmin

            # flip head_bbox
            xmin = head_bbox[0]
            xmax = head_bbox[2]
            head_bbox[0] = 1 - xmax
            head_bbox[2] = 1 - xmin



        # generate directed gaze_field using head loc and item loc
        direction = np.array(gaze) - np.array(eye)
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0
        direction = direction / norm
        direction = torch.from_numpy(direction)
        direction = torch.unsqueeze(direction, 0)
        # print('d',direction)
        # print('g',gaze)
        # print('e',eye)

        channel, height, width = gaze_field.shape
        gaze_field_directed = torch.from_numpy(np.copy(gaze_field))
        gaze_field_directed = torch.unsqueeze(gaze_field_directed, 0)
        gaze_field_directed = gaze_field_directed.permute([0, 2, 3, 1])
        gaze_field_directed = gaze_field_directed.view([1, -1, 2])



        gaze_field_directed = torch.matmul(gaze_field_directed, direction.view([1, 2, 1]).float())
        gaze_field_directed = gaze_field_directed.view([1, height, width, 1])
        gaze_field_directed = gaze_field_directed.permute([0, 3, 1, 2])
        # gaze_field_directed = torch.squeeze(gaze_field_directed, 0)
        gaze_field_directed[gaze_field_directed<0] = 0

        # print(torch.max(gaze_field_directed))
        # print(torch.min(gaze_field_directed))

        # different alpha
        gaze_field_directed_2 = torch.pow(gaze_field_directed, 2)
        gaze_field_directed_3 = torch.pow(gaze_field_directed, 5)
        gaze_field_directed_4 = torch.pow(gaze_field_directed, 20)
        gaze_field_directed_5 = torch.pow(gaze_field_directed, 100)
        gaze_field_directed_6 = torch.pow(gaze_field_directed, 500)

        directed_gaze_field = torch.cat([gaze_field_directed, gaze_field_directed_2,
                                         gaze_field_directed_3, gaze_field_directed_4,
                                         gaze_field_directed_5, gaze_field_directed_6])
        # gaze_field_directed = gaze_field_directed.numpy()



        # gt_bboxes = np.copy(data['ann']['bboxes']).tolist()
        # # gt_bboxes = np.expand_dims(gt_bboxes, axis=0)
        # gt_labels = np.copy(data['ann']['labels']).tolist()
        # # gt_labels = np.expand_dims(gt_labels, axis=0)
        # # print("boxes", gt_bboxes.shape)
        # # print("labels",gt_labels.shape)
        # print("boxes", len(gt_bboxes))
        # print("labels",len(gt_labels))

        # create bbox masks

        c,h,w = image.shape
        gaze_bbox_mask = np.zeros((h, w))
        gaze_bbox_mask[int(gaze_bbox[1]*h):int(gaze_bbox[3]*h),
                        int(gaze_bbox[0]*w):int(gaze_bbox[2]*w)] = 1
        gaze_bbox_mask = cv2.resize(gaze_bbox_mask, (224,224))
        gaze_bbox = gaze_bbox.astype(np.float32)

        head_bbox_mask = np.zeros((h, w))
        head_bbox_mask[int(head_bbox[1]*h):int(head_bbox[3]*h),
                        int(head_bbox[0]*w):int(head_bbox[2]*w)] = 1
        head_bbox_mask = cv2.resize(head_bbox_mask, (224,224))
        head_bbox = head_bbox.astype(np.float32)

        # Create GazeMask heatmap
        seg = data['seg']
        if seg is not None:
            seg_mask = create_mask(np.array(seg).astype(np.int64))
            seg_mask = cv2.resize(seg_mask, (224//4, 224//4))

            if flip_flag:
                seg_mask = cv2.flip(seg_mask, 1)    # horizontal flip

            seg_mask = seg_mask.astype(np.float64)/255.0
            if self.alpha < 1.0:
                gaze_mask = self.alpha * seg_mask + (1 - self.alpha) * heatmap
            elif self.alpha == 1.0:
                gaze_mask = np.fmax(seg_mask, heatmap)

        else:
            gaze_mask = np.zeros((224//4, 224//4))

        sample = {'image': image,
                  'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),   # this gaze field does not have direction yet
                                                                # direction is computed during training
                                                                # "ground truth" directed gaze field is not used
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0),
                  'image_path': image_path,

                  'original_image': original_image,
                  'gaze_idx': gaze_idx,
                  'gaze_class': gaze_class,
                  'gaze_bbox': gaze_bbox,
                  'head_bbox': head_bbox,
                  'head_loc': eye,
                  'directed_gaze_field': directed_gaze_field,
                  # 'gt_bboxes': gt_bboxes,
                  # 'gt_labels': gt_labels,
                  'gaze_bbox_mask': gaze_bbox_mask,
                  'head_bbox_mask': head_bbox_mask,

                  'gaze_mask_heatmap': torch.FloatTensor(gaze_mask).unsqueeze(0),        # GazeMask
                  'gaze_heatmap': torch.FloatTensor(heatmap).unsqueeze(0),  # original gaze heatmap
                  }


        if self.use_gazemask:
            sample['gt_heatmap'] = torch.FloatTensor(gaze_mask).unsqueeze(0)

        return sample

def create_mask(seg_idx, width=640, height=480):
    seg_idx = seg_idx.astype(np.int64)
    seg_mask = np.zeros((height,width)).astype(np.uint8)
    for i in range(seg_idx.shape[0]):
        seg_mask[seg_idx[i,1],seg_idx[i,0]] = 255
    return seg_mask


class GazeDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train', include_path=False):
        assert (training in set(['train', 'test']))
        print("Initializing Gaze Dataset...")
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training
        self.include_path = include_path

        anns = loadmat(self.mat_file)
        self.bboxes = anns[self.training + '_bbox']
        self.gazes = anns[self.training + '_gaze']
        self.paths = anns[self.training + '_path']
        self.eyes = anns[self.training + '_eyes']
        self.meta = anns[self.training + '_meta']
        self.image_num = self.paths.shape[0]

        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.paths[idx][0][0]
        image_path = os.path.join(self.root_dir, image_path)

        box = np.copy(self.bboxes[0, idx][0])
        # Note: original box annotations are [xmin, ymin, width, height]
        box[2] = box[0] + box[2] # xmax = xmin + width
        box[3] = box[1] + box[3] # ymax = xmax + height
        eye = self.eyes[0, idx][0]
        # todo: process gaze differently for training or testing
        gaze = self.gazes[0, idx].mean(axis=0)

        image_path = image_path.replace('\\', '/')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if random.random() > 0.5 and self.training == 'train':
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)

            xmin = box[0]
            xmax = box[2]
            box[0] = 1 - xmax
            box[2] = 1 - xmin

        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
        h, w = image.shape[:2]
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)

        head_bbox = np.array([x_0, y_0, x_1, y_1])
        # process image for saliency net
        # image = image_preprocess(image)
        original_image = np.copy(image)
        original_image = cv2.resize(original_image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        '''
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0
        direction = direction / norm
        '''

        # generate directed gaze_field using head loc and item loc
        direction = np.array(gaze) - np.array(eye)
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0
        direction = direction / norm
        direction = torch.from_numpy(direction)
        direction = torch.unsqueeze(direction, 0)
        # print('d',direction)
        # print('g',gaze)
        # print('e',eye)

        channel, height, width = gaze_field.shape
        gaze_field_directed = torch.from_numpy(np.copy(gaze_field))
        gaze_field_directed = torch.unsqueeze(gaze_field_directed, 0)
        gaze_field_directed = gaze_field_directed.permute([0, 2, 3, 1])
        gaze_field_directed = gaze_field_directed.view([1, -1, 2])

        gaze_field_directed = torch.matmul(gaze_field_directed, direction.view([1, 2, 1]).float())
        gaze_field_directed = gaze_field_directed.view([1, height, width, 1])
        gaze_field_directed = gaze_field_directed.permute([0, 3, 1, 2])
        # gaze_field_directed = torch.squeeze(gaze_field_directed, 0)
        gaze_field_directed[gaze_field_directed < 0] = 0

        # print(torch.max(gaze_field_directed))
        # print(torch.min(gaze_field_directed))

        # different alpha
        gaze_field_directed_2 = torch.pow(gaze_field_directed, 2)
        gaze_field_directed_3 = torch.pow(gaze_field_directed, 5)
        gaze_field_directed_4 = torch.pow(gaze_field_directed, 20)
        gaze_field_directed_5 = torch.pow(gaze_field_directed, 100)
        gaze_field_directed_6 = torch.pow(gaze_field_directed, 500)

        directed_gaze_field = torch.cat([gaze_field_directed, gaze_field_directed_2,
                                         gaze_field_directed_3, gaze_field_directed_4,
                                         gaze_field_directed_5, gaze_field_directed_6])

        # create bbox masks

        head_bbox_mask = np.zeros((h, w))
        head_bbox_mask[int(box[1] * h):int(box[3] * h),
        int(box[0] * w):int(box[2] * w)] = 1
        head_bbox_mask = cv2.resize(head_bbox_mask, (224,224))
        box = box.astype(np.float32)

        sample = {'image': image,
                  # 'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),     # this gaze field does not have direction yet
                                                                  # direction is computed during training
                                                                  # "ground truth" directed gaze field is not used
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0),
                  'image_path': image_path,

                  'original_image': original_image,
                  # 'gaze_idx': gaze_idx,       # not available in this dataset
                  'gaze_class': -1,           # not available in this dataset
                  # 'gaze_bbox': gaze_bbox,     # not available in this dataset


                  'head_bbox': torch.FloatTensor(head_bbox),
                  'head_loc': eye,
                  'directed_gaze_field': directed_gaze_field,
                  # 'gt_bboxes': gt_bboxes,             # not available in this dataset
                  # 'gt_labels': gt_labels,             # not available in this dataset
                  # 'gaze_bbox_mask': gaze_bbox_mask,   # not available in this dataset
                  'head_bbox_mask': head_bbox_mask,
                  }

        return sample
