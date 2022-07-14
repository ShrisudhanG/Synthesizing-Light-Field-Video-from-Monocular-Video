import torch
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable
import imageio

def imtensor2imnp(tensor_list):
    '''
    This will be of the size [N,3,h,w]
    returns a list of N images
    '''
    imnp = tensor_list
    list_im = [imnp[k].permute([0,2,3,1]).data.cpu().squeeze().numpy() for k in range(len(imnp))]
    return list_im

def lftensor2lfnp(lftensor_list):
    '''
    This will be of the size [N,V,3,h,w]
    returns a list of N LF images
    '''
    imnp = lftensor_list
    list_im = [imnp[k].permute([0,1,3,4,2]).data.cpu().squeeze().numpy() for k in range(len(imnp))]
    return list_im

def save_video_from_lf(lf_img, save_lf_path):
    h,w = lf_img.shape[-3:-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        save_lf_path, fourcc, 5, (w, h))
    for k in range(len(lf_img)):
        #print(lf_img[k, ..., ::-1].shape)
        out.write(np.uint8(lf_img[k, ..., ::-1] * 255))
    out.release()

def save_img(img, path):
    h,w = lf_img.shape[-2:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        save_lf_path, fourcc, 5, (w, h))
    for k in range(len(lf_img)):
        out.write(np.uint8(lf_img[k, ..., ::-1] * 255))
    out.release()

def get_paths(save_path, step, N):
    # get lf image paths
    pred_lf_paths = []
    ref_lf_paths = []
    img_paths = []
    gt_paths = []
    os.makedirs(os.path.join(save_path, f'seq{step:03d}'), exist_ok=True)
    for k in range(N):
        pred_lf_paths.append(os.path.join(save_path, f'seq{step:03d}/pred_lf_{k:03d}.mp4'))
        ref_lf_paths.append(os.path.join(save_path, f'seq{step:03d}/ref_lf_{k:03d}.mp4'))
        gt_paths.append(os.path.join(save_path, f'seq{step:03d}/gt_lf_{k:03d}.mp4'))
        img_paths.append(os.path.join(save_path, f'seq{step:03d}/inp_img_{k:03d}.jpg'))
    return gt_paths, pred_lf_paths, ref_lf_paths, img_paths