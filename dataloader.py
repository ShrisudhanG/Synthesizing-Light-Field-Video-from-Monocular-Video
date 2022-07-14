# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
import cv2
import skimage
import skimage.transform
from skimage import exposure
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as ttf
import torch.nn.functional as F

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, size, get_type='resize'):
    return transforms.Compose([ToTensor(mode=mode, size=size, get_type=get_type)])


class LFDataLoader(object):
    def __init__(self, args, mode):
        if args.type not in ['resize', 'crop']:
            raise ValueError('Get type should be either \'resize\' or \'crop\'')
            
        size = (args.height, args.width)
        self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode, size))
        self.data = DataLoader(self.testing_samples, args.batchsize,
                               shuffle=False, num_workers=1,
                               pin_memory=False, sampler=None)

        

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.type = args.type
        self.mode = mode
        self.transform = transform
        self.color_corr = args.color_corr


    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        paths = sample_path.split('\t')[:-1]
        sample = {}

        for i, path in enumerate(paths):
            lf_path = os.path.join(self.args.lf_path, path)
            lf = np.load(lf_path) / 255.0
            if lf.shape[4] == 3:
                lf = lf.transpose([0, 1, 4, 2, 3])
            else:
                lf = lf.transpose([1, 0, 2, 3, 4])

            if self.color_corr:
                mean = lf.mean()
                fact = np.log(0.4) / np.log(mean)
                if fact<1:
                    lf = lf ** fact

            X, Y, C, H, W = lf.shape
            image = lf[X//2, Y//2, ...]
            lf = lf.reshape(X*Y, C, H, W)
            
            disp_path = os.path.join(self.args.disp_path, path).replace('npy', 'png')
            disp = np.array(Image.open(disp_path)) / 255.0
            
            sample[i] = {'image': image, 'lf': lf, 'disp': disp}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, size, get_type):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Resize(size)
        self.type = get_type


    def __call__(self, sample):
        for i in sample.keys():
            image = sample[i]['image']
            image = self.to_tensor(image)
            if self.type == 'resize':
                image = self.transform(image)
            image = self.normalize(image)

            lf = sample[i]['lf']
            lf = self.to_tensor(lf)
            if self.type == 'resize':
                lf = self.transform(lf)
            
            disp = sample[i]['disp']
            disp = self.to_tensor(disp).unsqueeze(0)
            if self.type == 'resize':
                disp = self.transform(disp)

            sample[i] = {'image': image, 'lf': lf, 'disp': disp}
            
        return sample


    def to_tensor(self, pic):
        image = torch.FloatTensor(pic)
        
        shape = image.shape
        if len(shape) == 3 and shape[-1] == 3:
            image = image.permute(2, 0, 1)
        elif len(shape) == 4 and shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        return image
