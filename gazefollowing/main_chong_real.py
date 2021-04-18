import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.chong import ModelSpatial
# from dataset import GazeFollowOrig
from dataloader.chong import GazeDataset, GooDataset
from dataloader import chong_imutils
from training.train_chong import train, test, GazeOptimizer

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
# from scipy.misc import imresize
# from tensorboardX import SummaryWriter
# import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import cv2

from utils_logging import setup_logger
from models.__init__ import save_checkpoint, resume_checkpoint


from parse_inputs import parse_inputs
args = parse_inputs()


logger = setup_logger(name='first_logger',
                      log_dir ='./logs/',
                      log_file=args.log_file,
                      log_format = '%(asctime)s %(levelname)s %(message)s',
                      verbose=True)

def main():
    # transform = _get_transform(args.input_resolution)

    # Prepare data
    print("Loading Data")

    batch_size = args.batch_size
    train_set = GooDataset(args.train_dir, args.train_annotation, 'train')
    train_data_loader = DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)

    if args.test_dir is not None:
        print('==> Loading Test Dataset')
        test_set = GooDataset(args.test_dir, args.test_annotation, 'test')
        test_data_loader = DataLoader(test_set, batch_size=batch_size//2,
                                    shuffle=False, num_workers=8)

    # Loads model
    print("Constructing model")
    net = ModelSpatial()
    net.cuda()
    # net.cuda().to(device)

    # Hyperparameters
    start_epoch = 25
    max_epoch = 26
    learning_rate = args.init_lr

    # Initial weights chong
    if args.init_weights:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(args.init_weights)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    # Initializes Optimizer
    gaze_opt = GazeOptimizer(net, learning_rate)
    optimizer = gaze_opt.getOptimizer(start_epoch)

    # Resuming Training
    resume_training = args.resume_training
    print(resume_training)
    if resume_training:
        net, optimizer, start_epoch = resume_checkpoint(net, optimizer, args.resume_path)
        if args.test_dir is not None:
            test(net, test_data_loader,logger)

    for epoch in range(start_epoch, max_epoch):

        # Update optimizer
        optimizer = gaze_opt.getOptimizer(epoch)

        # Train model
        train(net, train_data_loader, optimizer, epoch, logger)

        # # Save model and optimizer at the last 5 epochs
        # if epoch == 25:
        save_path = './saved_models/real_chong/'
        save_checkpoint(net, optimizer, epoch+1, save_path)

        # Evaluate model
        if args.test_dir is not None:
            test(net, test_data_loader, logger)



if __name__ == "__main__":
    main()
