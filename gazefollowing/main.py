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
from models.__init__ import save_checkpoint, resume_checkpoint 

from parse_inputs import parse_inputs
args = parse_inputs()

if args.baseline == 'recasens':
    from models.recasens import GazeNet
    from dataloader.recasens import GooDataset, GazeDataset
    from training.train_recasens import train, test, GazeOptimizer
elif args.baseline == 'gazenet':
    from models.gazenet import GazeNet   
    from dataloader.gazenet import GooDataset, GazeDataset
    from training.train_gazenet import train, test, GazeOptimizer
else:
    print('Model %s does not exist. Defaulting to gazenet.' % (args.baseline))
    from models.gazenet import GazeNet   
    from dataloader.gazenet import GooDataset, GazeDataset
    from training.train_gazenet import train, test, GazeOptimizer

logger = setup_logger(name='first_logger', 
                      log_dir ='./logs/',
                      log_file=args.log_file,
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)
def main():

    # Dataloaders for GOO
    batch_size = args.batch_size

    print('==> Loading Train Dataset')
    train_set = GooDataset(args.train_dir, args.train_annotation, 'train', use_gazemask=args.gazemask)
    train_data_loader = DataLoader(train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=16)
    
    if args.test_dir is not None:
        print('==> Loading Test Dataset')
        test_set = GooDataset(args.test_dir, args.test_annotation, 'test')
        test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                                    shuffle=False, num_workers=8)

    # Loads model
    net = GazeNet()
    net.cuda()

    # Hyperparameters
    start_epoch = 0
    max_epoch = 25
    learning_rate = args.init_lr

    # Initializes Optimizer
    gaze_opt = GazeOptimizer(net, learning_rate)
    optimizer = gaze_opt.getOptimizer(start_epoch)

    # Resuming Training
    resume_training = args.resume_training
    if resume_training:
        net, optimizer, start_epoch = resume_checkpoint(net, optimizer, args.resume_path)       
        if args.test_dir is not None: 
            test(net, test_data_loader,logger)

    for epoch in range(start_epoch, max_epoch):
        
        # Update optimizer
        optimizer = gaze_opt.getOptimizer(epoch)

        # Train model
        train(net, train_data_loader, optimizer, epoch, logger)

        # Save model and optimizer at the last 5 epochs
        if epoch > max_epoch-5:
            save_path = args.save_model_dir
            save_checkpoint(net, optimizer, epoch+1, save_path)
        
        # Evaluate model
        if args.test_dir is not None:
            test(net, test_data_loader, logger)
    
if __name__ == "__main__":
    main()             
