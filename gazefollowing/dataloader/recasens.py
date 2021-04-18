from __future__ import print_function, division
import pickle
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.patches as patches
import scipy.io as sio
from .customtransforms import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop

#ADR stuff
#from dataset_domrand import DomainRandomizer

def parse_GooPickle(images_dir, pickle_path):

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        num_data = len(data)

        path, boxes, points, eyes = ([] for i in range(4))

        for i in range(num_data):

            box = np.asarray(data[i]['ann']['bboxes'][-1,:]).astype(np.float)
            #print(float(data[i]['gaze_cx']))
            point = np.array([data[i]['gaze_cx'],data[i]['gaze_cy']]).astype(np.float)
            eye = np.array([data[i]['hx'],data[i]['hy']]).astype(np.float)

            #normalize
            box = box / [640, 480, 640, 480]
            point = point / [640, 480]
            eye = eye / [640, 480]
            full_path = images_dir + data[i]['filename'].replace('\\', '/')


            path.append(full_path)
            boxes.append(box)
            points.append(point)
            eyes.append(eye)

    boxes = np.asarray(boxes)
    points = np.asarray(points)
    eyes = np.asarray(eyes)

    print('Total images in this set:', num_data)
    #print(path)
    return boxes, points, eyes, path

def getCropped(img, e):

    alpha = 0.3
    w_x = int(math.floor(alpha * img.shape[1]))
    w_y = int(math.floor(alpha * img.shape[0]))

    if w_x % 2 == 0:
        w_x = w_x + 1

    if w_y % 2 == 0:
        w_y = w_y + 1

    im_face = np.ones((w_y, w_x, 3))
    im_face[:, :, 0] = 123 * np.ones((w_y, w_x))
    im_face[:, :, 1] = 117 * np.ones((w_y, w_x))
    im_face[:, :, 2] = 104 * np.ones((w_y, w_x))

    center = [math.floor(e[0] * img.shape[1]), math.floor(e[1] * img.shape[0])]
    d_x = math.floor((w_x - 1) / 2)
    d_y = math.floor((w_y - 1) / 2)

    bottom_x = center[0] - d_x - 1
    delta_b_x = 0
    if bottom_x < 0:
        delta_b_x = 1 - bottom_x
        bottom_x = 0

    top_x = center[0] + d_x - 1
    delta_t_x = w_x - 1
    if top_x > img.shape[1] - 1:
        delta_t_x = w_x - (top_x - img.shape[1] + 1)
        top_x = img.shape[1] - 1

    bottom_y = center[1] - d_y - 1
    delta_b_y = 0
    if bottom_y < 0:
        delta_b_y = 1 - bottom_y
        bottom_y = 0

    top_y = center[1] + d_y - 1
    delta_t_y = w_y - 1
    if top_y > img.shape[0] - 1:
        delta_t_y = w_y - (top_y - img.shape[0] + 1)
        top_y = img.shape[0] - 1


    x = img[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1), :]
    x = np.ascontiguousarray(x)

    try:
        x = transform.resize(x,(227, 227))
        return x
    except:
        # print('begins') #TODO, are there corrupt images?
        # print(x)
        #print(x.shape)
        # print('ends')
        print('corruption')
        return img

class GooDataset(Dataset):

    def __init__(self, images_path, pickle_path, type):

        data_bbox, data_gaze, data_eyes, data_path = parse_GooPickle(images_path, pickle_path)

        self.bbox_list = data_bbox
        self.gaze_list = data_gaze
        self.eyes_list = data_eyes
        self.img_path_list = data_path
        self.type = type

    def __len__(self):
        len_images = len(self.img_path_list)

        return  len_images

    def __getitem__(self, idx):

        img_name = self.img_path_list[idx]
        img = io.imread(img_name)
        try:
            img = color.rgba2rgb(img)
        except:
            pass

        s = img.shape
        #print('img_orig_shape: ', s)
        bbox_corr = self.bbox_list[idx]
        bbox_corr[bbox_corr < 0] = 0.0
        bbox = np.copy(img[ int(bbox_corr[1] * s[0]): int(np.ceil( bbox_corr[1] * s[0] + bbox_corr[3] * s[0])), int(bbox_corr[0] * s[1]): int(np.ceil(bbox_corr[0] * s[1] + bbox_corr[2] * s[1]))])

        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        img = np.ascontiguousarray(img)
        img = transform.resize(img,(227, 227))
        #print('after resize: ',img.shape)

        gaze = self.gaze_list[idx]

        eyes = self.eyes_list[idx]
        eyes_bbox = (eyes - bbox_corr[:2])/bbox_corr[2:]

        if len(img.shape) == 2:
            #print('stacking')
            img = np.stack((img,)*3, axis=-1)

        if len(bbox.shape) == 2:
            #print('stacking bbox')
            bbox = np.stack((bbox,) * 3, axis=-1)

        bbox = getCropped(bbox, eyes_bbox)
        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        eyes_loc_size = 13
        gaze_label_size = 5

        ######DO DATA AUG HERE###########

        if self.type == 'train':

            composed = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), RandomCrop()])
            sample = {'img': img, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}
            sample = composed(sample)

            img = sample['img']
            bbox = sample['bbox']
            eyes = sample['eyes']
            eyes_bbox = sample['eyes_bbox']
            gaze = sample['gaze']

        #SHIFTED GRIDS STUFF

        eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
        eyes_loc[int(np.floor(eyes_loc_size * eyes[1]))][int(np.floor(eyes_loc_size * eyes[0]))] = 1

        grid_size = 5

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        x_pix = gaze[0]
        y_pix = gaze[1]

        shifted_grids = np.zeros((grid_size, gaze_label_size, gaze_label_size))
        for i in range(5):

            x_grid = int(np.floor( gaze_label_size * gaze[0] + (v_x[i] * (1/ (grid_size * 3.0))) ) )
            y_grid = int(np.floor( gaze_label_size * gaze[1] + (v_y[i] * (1/ (grid_size * 3.0))) ) )

            if x_grid < 0:
                x_grid = 0
            elif x_grid > 4:
                x_grid = 4
            if y_grid < 0:
                y_grid = 0
            elif y_grid > 4:
                y_grid = 4

            try:
                shifted_grids[i][y_grid][x_grid] = 1
            except:
                print("**************")
                print("eyes: ", eyes)
                print("eyes_bbox: ", eyes_bbox)
                print("gaze: ", gaze)
                print("grid vals: ", x_grid, y_grid)
                plt.imshow(img)
                plt.savefig('a.jpg')
                exit()

        eyes_loc = torch.from_numpy(eyes_loc).contiguous()
        shifted_grids = torch.from_numpy(shifted_grids).contiguous()

        shifted_grids = shifted_grids.view(1, 5, 25)
        gaze_final = np.ones(100)
        gaze_final *= -1
        gaze_final[:gaze.shape[0]] = gaze

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).contiguous()

        bbox = bbox.transpose((2, 0, 1))
        bbox = torch.from_numpy(bbox.copy()).contiguous()

        normtransform = transforms.Compose([
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

        #print(img.shape)
        img = normtransform(img.float())
        bbox = normtransform(bbox.float())

        sample = (img.float(), bbox.float(), eyes_loc.float(), shifted_grids.float(), eyes, idx, eyes_bbox, gaze_final)

        return sample

class GazeDataset(Dataset):

    def __init__(self, images_path, pickle_path, type):

        self.type = type
        path = images_path
        
        if type == 'train':
            Data = sio.loadmat(pickle_path)
        if type == 'test':
            Data = sio.loadmat(pickle_path)

        if type == 'train':
            data_bbox = Data['train_bbox'][0]
            data_eyes = Data['train_eyes'][0]
            data_gaze = Data['train_gaze'][0]
            data_meta = Data['train_meta'][0]
            data_path = Data['train_path']

        if type == 'test':
            data_bbox = Data['test_bbox'][0]
            data_eyes = Data['test_eyes'][0]
            data_gaze = Data['test_gaze'][0]
            data_meta = Data['test_meta'][0]
            data_path = Data['test_path']

        for i in range(data_path.shape[0]):
            data_path[i] = data_path[i][0]
        data_path = data_path.flatten()
        data_path = path + data_path

        for i in range(data_bbox.shape[0]):
            data_bbox[i] = data_bbox[i].flatten()

        for i in range(data_eyes.shape[0]):
            data_eyes[i] = data_eyes[i].flatten()

        for i in range(data_gaze.shape[0]):
            data_gaze[i] = data_gaze[i].flatten()

        self.img_path_list = data_path
        self.bbox_list = data_bbox
        self.eyes_list = data_eyes
        self.gaze_list = data_gaze

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img = io.imread(img_name)

        s = img.shape
        bbox_corr = self.bbox_list[idx]
        bbox_corr[bbox_corr < 0] = 0.0
        bbox = np.copy(img[ int(bbox_corr[1] * s[0]): int(np.ceil( bbox_corr[1] * s[0] + bbox_corr[3] * s[0])), int(bbox_corr[0] * s[1]): int(np.ceil(bbox_corr[0] * s[1] + bbox_corr[2] * s[1]))])

        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        img = np.ascontiguousarray(img)
        img = transform.resize(img,(227, 227))

        gaze = self.gaze_list[idx]

        eyes = self.eyes_list[idx]
        eyes_bbox = (eyes - bbox_corr[:2])/bbox_corr[2:]

        if len(img.shape) == 2:
            #print('stacking')
            img = np.stack((img,)*3, axis=-1)

        if len(bbox.shape) == 2:
            #print('stacking bbox')
            bbox = np.stack((bbox,) * 3, axis=-1)

        bbox = getCropped(bbox, eyes_bbox)
        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        eyes_loc_size = 13
        gaze_label_size = 5

        ######DO DATA AUG HERE###########

        if self.type == 'train':

            composed = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), RandomCrop()])
            sample = {'img': img, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}
            sample = composed(sample)

            img = sample['img']
            bbox = sample['bbox']
            eyes = sample['eyes']
            eyes_bbox = sample['eyes_bbox']
            gaze = sample['gaze']

        #SHIFTED GRIDS STUFF

        eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
        eyes_loc[int(np.floor(eyes_loc_size * eyes[1]))][int(np.floor(eyes_loc_size * eyes[0]))] = 1

        grid_size = 5

        v_x = [0, 1, -1, 0, 0]
        v_y = [0, 0, 0, -1, 1]

        x_pix = gaze[0] 
        y_pix = gaze[1] 

        shifted_grids = np.zeros((grid_size, gaze_label_size, gaze_label_size)) 
        for i in range(5):

            x_grid = int(np.floor( gaze_label_size * gaze[0] + (v_x[i] * (1/ (grid_size * 3.0))) ) ) 
            y_grid = int(np.floor( gaze_label_size * gaze[1] + (v_y[i] * (1/ (grid_size * 3.0))) ) )
 
            if x_grid < 0:
                x_grid = 0
            elif x_grid > 4:
                x_grid = 4
            if y_grid < 0:
                y_grid = 0
            elif y_grid > 4:
                y_grid = 4

            try:
                shifted_grids[i][y_grid][x_grid] = 1
            except:
                print("**************")
                print("eyes: ", eyes)
                print("eyes_bbox: ", eyes_bbox)
                print("gaze: ", gaze)
                print("grid vals: ", x_grid, y_grid)
                plt.imshow(img)
                plt.savefig('a.jpg')
                exit()

        eyes_loc = torch.from_numpy(eyes_loc).contiguous()
        shifted_grids = torch.from_numpy(shifted_grids).contiguous()
        
        shifted_grids = shifted_grids.view(1, 5, 25)
        gaze_final = np.ones(100) 
        gaze_final *= -1
        gaze_final[:gaze.shape[0]] = gaze  

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).contiguous()

        bbox = bbox.transpose((2, 0, 1))
        bbox = torch.from_numpy(bbox.copy()).contiguous()

        normtransform = transforms.Compose([
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

        img = normtransform(img.float())
        bbox = normtransform(bbox.float())

        sample = (img.float(), bbox.float(), eyes_loc.float(), shifted_grids.float(), eyes, idx, eyes_bbox, gaze_final)

        return sample