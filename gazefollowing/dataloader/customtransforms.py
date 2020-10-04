import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class RandomHorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        image = sample['img']
        bbox = sample['bbox']
        eyes = sample['eyes']
        eyes_bbox = sample['eyes_bbox']
        gaze = sample['gaze']

        h, w = image.shape[:2]
        h_bbox, w_bbox = bbox.shape[:2]

        if random.random() > 0.5:

            image = image[:, ::-1]
            bbox = bbox[:, ::-1]
            eyes[0] = 1 - eyes[0]
            eyes_bbox[0] = 1 - eyes_bbox[0]
            gaze[0] = 1 - gaze[0]

        sample = {'img': image, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}

        return sample

class RandomVerticalFlip(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        image = sample['img']
        bbox = sample['bbox']
        eyes = sample['eyes']
        eyes_bbox = sample['eyes_bbox']
        gaze = sample['gaze']

        h, w = image.shape[:2]
        h_bbox, w_bbox = bbox.shape[:2]

        if random.random() > 0.5:

            image = image[::-1, :]
            bbox = bbox[::-1, :]
            eyes[1] = 1 - eyes[1]
            eyes_bbox[1] = 1 - eyes_bbox[1]
            gaze[1] = 1 - gaze[1]

        sample = {'img': image, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}

        return sample

class RandomCrop(object): 

    def __init__(self):

        self.scale_size = (240, 240)
        self.orig_size = (227, 227)

    def __call__(self, sample):

        image = sample['img']
        bbox = sample['bbox']
        eyes = sample['eyes']
        eyes_bbox = sample['eyes_bbox']
        gaze = sample['gaze']

        h, w = image.shape[:2]

        scale_h, scale_w = self.scale_size
        new_h, new_w = self.orig_size

        image = transform.resize(image, (scale_h, scale_w))

        top = np.random.randint(0, scale_h - new_h)
        left = np.random.randint(0, scale_w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        eyes = (eyes * 227 - [left, top]) / (1.0 * 227)   ##assuming that w is equal to h
        eyes_bbox = (eyes_bbox * 227- [left, top]) / (1.0 * 227)
        gaze = (gaze * 227 - [left, top]) / (1.0 * 227)

        if eyes[0] < 0:
            eyes[0] = 0.05
        if eyes[1] < 0:
            eyes[1] = 0.05
        if eyes[0] > 1:
            eyes[0] = 0.95
        if eyes[1] > 1:
            eyes[1] = 0.95

        if eyes_bbox[0] < 0:
            eyes_bbox[0] = 0.05
        if eyes_bbox[1] < 0:
            eyes_bbox[1] = 0.05
        if eyes_bbox[0] > 1:
            eyes_bbox[0] = 0.95
        if eyes_bbox[1] > 1:
            eyes_bbox[1] = 0.95

        if gaze[0] < 0:
            gaze[0] = 0.05
        if gaze[1] < 0:
            gaze[1] = 0.05
        if gaze[0] > 1:
            gaze[0] = 0.95
        if gaze[1] > 1:
            gaze[1] = 0.95

        sample = {'img': image, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}

        return sample
