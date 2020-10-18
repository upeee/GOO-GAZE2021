import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

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
from utils import get_paste_kernel, kernel_map
from utils_logging import setup_logger


from models.gazenet import GazeNet
from models.__init__ import save_checkpoint, resume_checkpoint
from dataloading import GazeDataset
from training.train_gazenet import train, test
from training.train_gazenet import StagedOptimizer

logger = setup_logger(name='first_logger', 
                      log_dir ='./logs/',
                      log_file='train.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

def main():

    # Load Datasets, Initialize DataLoaders
    batch_size = args.batch_size

    # root_dir = '/home/eee198/Documents/datasets/GazeFollowData/',
    # mat_file = '/home/eee198/Documents/datasets/GazeFollowData/train_annotations.mat',

    train_set = GazeDataset(root_dir=args.train_root_dir,
                            mat_file=args.train_mat_file,
                            training='train')
    train_data_loader = DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=16)

    # root_dir = '/home/eee198/Documents/datasets/GazeFollowData/',
    # mat_file = '/home/eee198/Documents/datasets/GazeFollowData/test_annotations.mat',


    if args.test_root_dir is not None:
        test_set = GazeDataset(root_dir=args.test_root_dir,
                                    mat_file=args.test_mat_file,
                                    training='test')
        test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                                    shuffle=False, num_workers=8)

    #Initialized Models
    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    #Configs
    start_epoch = 0
    method = 'Adam'
    learning_rate = 0.0001
    max_epoch = 25

    #Initialize StagedOptimizer (GazeNet only)
    staged_opt = StagedOptimizer(net, learning_rate)

    #Is training resumed from previous run?
    resume_training = False
    resume_path = './saved_models/temp/model_epoch25.pth.tar'
    if resume_training :
        net, optimizer = resume_checkpoint(net, optimizer=None, resume_path=resume_path)
        test(net, test_data_loader,logger)
        start_epoch = 25

    for epoch in range(start_epoch, max_epoch):
        
        # Update optimizer
        optimizer = staged_opt.update(epoch)

        # Train model
        train(net, train_data_loader, optimizer, epoch, logger)

        # Save model and optimizer
        if epoch > max_epoch-5:
            save_path = './saved_models/temp/'
            save_checkpoint(net, optimizer, epoch+1, save_path)
        
        # Evaluate model
        if args.test_root_dir is not None:
            with torch.no_grad():
                test(net, test_data_loader, logger)

if __name__ == "__main__":
    main()