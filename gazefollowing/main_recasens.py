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

from models.recasens import GazeNet
from dataloader.recasens import GooDataset, GazeDataset
from models.__init__ import save_checkpoint, resume_checkpoint
from training.train_recasens import train, test, GazeOptimizer

logger = setup_logger(name='first_logger', 
                      log_dir ='./logs/',
                      log_file='train_recasens.log',
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

def main():

    # Dataloaders for GOO
    batch_size=32
    workers=12
    testbatchsize=32

    images_dir = '/hdd/HENRI/goosynth/1person/GazeDatasets/'
    pickle_path = '/hdd/HENRI/goosynth/picklefiles/trainpickle2to19human.pickle'
    test_images_dir = '/hdd/HENRI/goosynth/test/'
    test_pickle_path = '/hdd/HENRI/goosynth/picklefiles/testpickle120.pickle'

    train_set = GooDataset(images_dir, pickle_path, 'train')
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)

    val_set = GooDataset(test_images_dir, test_pickle_path, 'test')
    test_data_loader = torch.utils.data.DataLoader(val_set, batch_size=testbatchsize, num_workers=workers, shuffle=False)

    # Loads model
    net = GazeNet(placesmodel_path='./alexnet_places365.pth')
    net.cuda()

    start_epoch = 0
    max_epoch = 5
    learning_rate = 0.0001

    resume_training = False
    resume_path = './saved_models/temp/gazenet_gazefollow_0.1647_11epoch_0modIdx.pth.tar'
    if resume_training :
        net, optimizer = resume_checkpoint(net, optimizer, resume_path)
        test(net, test_data_loader,logger)
        start_epoch = 25

    staged_opt = GazeOptimizer(net, learning_rate)

    for epoch in range(start_epoch, max_epoch):
        
        # Update optimizer
        optimizer = staged_opt.getOptimizer(epoch)

        # Train model
        train(net, train_data_loader, optimizer, epoch, logger)

        # Save model and optimizer
        if epoch > max_epoch-5:
            save_path = './saved_models/temp/'
            save_checkpoint(net, optimizer, epoch+1, save_path)
        
        # Evaluate model
        test(net, test_data_loader, logger)


if __name__ == "__main__":
    main()