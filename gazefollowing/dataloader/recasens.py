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
        #print('corruption')
        return img

class GooRealDataset(Dataset):

    def __init__(self, data_bbox, data_gaze, data_eyes, data_path, type, path, opt=None, domrand=None, domrand_ipp=60000):

        self.type = type
        if opt:
            self.opt = opt


        self.bbox_list = data_bbox
        self.gaze_list = data_gaze
        self.eyes_list = data_eyes
        self.img_path_list = data_path
        self.domrand = domrand

        if self.domrand:
            self.pickle_dict = {}

            for idx in range(len(self.img_path_list)):
                img_tag = os.path.basename(self.img_path_list[idx])
                img_tag = img_tag.replace('-img1.png', '')
                self.pickle_dict[img_tag] = idx

            self.plotnum = 0
            self.ipp = domrand_ipp #IMAGES_PER_PLOT, constant
            self.remaining_trained_imgs = domrand_ipp #initial value only

    def __len__(self):
        len_images = len(self.img_path_list)

        if self.domrand:
                totalImgsForCurrParams = self.domrand.calculateTotalImagesWithinBounds()
                #print("totalImgsForCurrParams: ", totalImgsForCurrParams)
                if self.ipp > totalImgsForCurrParams:
                    len_images = int(totalImgsForCurrParams)
                else:
                    len_images = self.ipp

        return  len_images

    def __getitem__(self, idx):

        if self.domrand:
            paramStrGenerated = self.domrand.generateImage()
            #print(paramStrGenerated)

            while not paramStrGenerated in self.pickle_dict:
                paramStrGenerated = self.domrand.generateImage()
            #print('success')
            idx = self.pickle_dict[paramStrGenerated]

        img_name = self.img_path_list[idx]
        img = io.imread(img_name)
        #img = color.rgba2rgb(img)

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
        img = color.rgba2rgb(img)

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

class GooDatasetADR(Dataset):

    def __init__(self, data_bbox, data_gaze, data_eyes, data_path, type, path, multi_model_adr, images_per_plot=60000):

        self.type = type
        
        self.bbox_list = data_bbox
        self.gaze_list = data_gaze
        self.eyes_list = data_eyes
        self.img_path_list = data_path
        
        #ADR-related
        self.multi_model_adr = multi_model_adr
        self.domrand_idx = 0
        self.total_rand_params = self.multi_model_adr.TOTAL_RAND_PARAMS
        self.ipp = images_per_plot
          
        self.pickle_dict = {}

        for idx in range(len(self.img_path_list)):
            img_tag = os.path.basename(self.img_path_list[idx])
            img_tag = img_tag.replace('-img1.png', '')
            self.pickle_dict[img_tag] = idx

    def __len__(self):
        len_images = len(self.img_path_list)

        domrand = self.multi_model_adr.dom_randomizers[self.domrand_idx]
        if domrand:
            len_images = self.ipp

        return  len_images
    
    def set_domrand(self, domrand_idx):
        
        if domrand_idx not in range(self.total_rand_params):
            print('Invalid domrand_idx. Current idx value %d is retained.' % self.domrand_idx)
        
        prev_value = self.domrand_idx
        self.domrand_idx = domrand_idx
        print("Domain Randomizer is set from %d --> %d." % (prev_value, self.domrand_idx))
        
    def get_domrand(self):
        
        print("Domain Randomizer # %d is currently used." % self.domrand_idx)
        return self.domrand_idx
        
    def __getitem__(self, idx):    
        
        domrand = self.multi_model_adr.dom_randomizers[self.domrand_idx]

        if domrand:

            paramStrGenerated = domrand.generateImage()
            #print(paramStrGenerated)

            while not paramStrGenerated in self.pickle_dict:
                paramStrGenerated = domrand.generateImage()
            #print('success')
            idx = self.pickle_dict[paramStrGenerated]

        img_name = self.img_path_list[idx]
        img = io.imread(img_name)
        img = color.rgba2rgb(img)

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

        sample = (img.float(), bbox.float(), eyes_loc.float(), shifted_grids.float(), \
                  eyes, idx, eyes_bbox, gaze_final)

        return sample

class GooLoader():

    def __init__(self, opt):



        self.train_gaze = GazeDataset(Train_Ann, 'train', opt.data_dir, opt)
        self.x = self.train_gaze[1]
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

        self.val_gaze = GazeDataset(Test_Ann, 'test', opt.data_dir, opt)
        self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        batch_size=opt.testbatchsize, shuffle=False, num_workers=opt.workers)

class GooLoader_manual():

    def __init__(self, opt=None, gooreal=False, oneshot=False):

        if not oneshot:
            images_dir = '/hdd/HENRI/goosynth/1person/GazeDatasets/'
            pickle_path = '/hdd/HENRI/goosynth/picklefiles/trainpickle2to19human.pickle'
        else:
            images_dir = '/home/eee198/Documents/datasets/GOOReal/finalrealdatasetImgsV2/'
            pickle_path = '/home/eee198/Documents/datasets/GOOReal/oneshotrealhumans.pickle'

        batch_size=32
        workers=12
        testbatchsize=32

        domrand = None
        if domrand:
            print('*****Auto-Domain-Randomization enabled.*****')

        boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(images_dir, pickle_path)

        if not oneshot:
            self.train_gaze = GooDataset(boxes_list, gaze_list, eyes_list, path_list, 'train', images_dir, domrand=domrand)
        else:
            self.train_gaze = GooRealDataset(boxes_list, gaze_list, eyes_list, path_list, 'train', images_dir, domrand=domrand)
            
        self.x = self.train_gaze[1]
        #print(len(self.train_gaze[1]))
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=batch_size, shuffle=True, num_workers=workers)

        if gooreal:
            test_images_dir = '/home/eee198/Documents/datasets/GOOReal/finalrealdatasetImgsV2/'
            test_pickle_path = '/home/eee198/Documents/datasets/GOOReal/testrealhumans.pickle'     
        else:
            test_images_dir = '/hdd/HENRI/goosynth/test/'
            test_pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'

        boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(test_images_dir, test_pickle_path)

        if gooreal:
            self.val_gaze = GooRealDataset(boxes_list, gaze_list, eyes_list, path_list, 'test', test_images_dir)
        else:
            self.val_gaze = GooDataset(boxes_list, gaze_list, eyes_list, path_list, 'test', test_images_dir)

        self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        batch_size=testbatchsize, num_workers=workers, shuffle=False)

class GooLoaderADR_manual():

    def __init__(self, opt=None):

        images_dir = '/hdd/HENRI/goosynth/1person/GazeDatasets/'
        pickle_path = '/hdd/HENRI/goosynth/picklefiles/trainpickle2to19human.pickle'
        #pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'
        batch_size=32
        workers=12
        testbatchsize=32

        #domrand = None
        multi_model_adr = MultiModelEval(TOTAL_RAND_PARAMS=5)
        
        for i in range(5):
            multi_model_adr.dom_randomizers[i] = DomainRandomizer(20000)

        boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(images_dir, pickle_path)

        self.train_gaze = GooDatasetADR(boxes_list, gaze_list, eyes_list, path_list, 'train', images_dir, multi_model_adr=multi_model_adr)
        self.x = self.train_gaze[1]
        #print(len(self.train_gaze[1]))
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=batch_size, shuffle=True, num_workers=workers)

        #test_images_dir = '/hdd/HENRI/goosynth/test/'
        #test_pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'

        #boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(test_images_dir, test_pickle_path)

        #self.val_gaze = GooDataset(boxes_list, gaze_list, eyes_list, path_list, 'test', test_images_dir)
        #self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        #batch_size=testbatchsize, num_workers=workers, shuffle=False)

class GooLoaderADR():

    def __init__(self, multi_model_adr, opt=None):

        images_dir = '/hdd/HENRI/goosynth/1person/GazeDatasets/'
        pickle_path = '/hdd/HENRI/goosynth/picklefiles/trainpickle2to19human.pickle'
        #pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'
        batch_size=32
        workers=12
        testbatchsize=32

        boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(images_dir, pickle_path)

        self.train_gaze = GooDatasetADR(boxes_list, gaze_list, eyes_list, path_list, 'train', images_dir, multi_model_adr=multi_model_adr)
        self.x = self.train_gaze[1]
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=batch_size, shuffle=True, num_workers=workers)

        test_images_dir = '/hdd/HENRI/goosynth/test/'
        test_pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'

        boxes_list, gaze_list, eyes_list, path_list = parse_GooPickle(test_images_dir, test_pickle_path)

        self.val_gaze = GooDataset(boxes_list, gaze_list, eyes_list, path_list, 'test', test_images_dir)
        self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        batch_size=testbatchsize, num_workers=workers, shuffle=False)