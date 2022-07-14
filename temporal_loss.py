import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import json
import numpy as np
import random
import math
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torchvision
from lpips_pytorch import LPIPS

import model_io
import models
from dataloader import LFDataLoader
from loss_temp import *
from utils import RunningAverage, RunningAverageDict, denormalize
import tensor_ops as utils


class Tester():
    def __init__(self, args):

        self.args = args
        #################################### Setup GPU device ######################################### 
        self.device = torch.device(f'cuda:{args.gpu_1}' if torch.cuda.is_available() else 'cpu')
        self.device1 = torch.device(f'cuda:{args.gpu_2}' if torch.cuda.is_available() else 'cpu')
        print('Device: {}, {}'.format(self.device, self.device1))
        
        #################################### Refinement Model #########################################
        self.ref_model = models.RefinementBlock(patch_size=1)
        checkpoint = torch.load('weights/refine_net.pt', map_location='cpu')['model']
        self.ref_model.load_state_dict(checkpoint)
        self.ref_model = self.ref_model.to(self.device)

        ####################################### LF Model ##############################################
        # number of predictions for TD
        td_chans = self.args.rank*self.args.num_layers*3
        self.lf_model = models.UnetLF.build(td_chans=td_chans, layers=args.num_layers, rank=args.rank)
        self.lf_model.encoder.original_model.conv_stem = models.Conv2dSame(10, 48, kernel_size=(3, 3), 
                                                                           stride=(2, 2), bias=False)
        checkpoint = torch.load('weights/recons_net.pt', map_location='cpu')['model']
        self.lf_model.load_state_dict(checkpoint)
        self.lf_model = self.lf_model.to(self.device1)

        ##################################### Tensor Display ##########################################
        self.val_td_model = models.multilayer(height=args.height, width=args.width, 
                                              args=self.args, device=self.device1)
        self.md = args.max_displacement
        self.zp = args.zero_plane

        ########################################## Losses #############################################
        self.temporal_loss = TemporalConsistency(args, self.device)
        
        ####################################### Save test results ##############################################
        self.save_path = os.path.join(args.results, args.dataset+'-{:.2f}, {:.2f}'.format(self.md, self.zp))
        os.makedirs(self.save_path, exist_ok=True)
        self.save_numpy = args.save_numpy


    def compute_losses(self, curr_lf, prev_lf, pred_lf):
        temp_loss = self.temporal_loss(curr_lf, prev_lf, pred_lf)
        return temp_loss


    def test(self, test_loader, md, zp):
        ###############################################################################################
        self.lf_model.eval()

        ###############################################################################################
        # some globals
        iters = len(test_loader)
        temp_loss_avg = RunningAverage()
        f = open(f'{self.save_path}/{self.args.dataset}({md:.1f},{zp:.1f})-temp_loss_ref.txt', 'w')

        ################################# Validation loop #############################################
        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), 
                      desc='Testing-{}_{:.2f},{:.2f}'.format(self.args.dataset, md, zp)) as vepoch:
                for i, batch in vepoch:
                    ids = sorted(batch.keys())
                    leng = len(ids) - 1
                    id = ids[len(ids)//2]
                    prev_state = None

                    pred_lfs = []
                    gt_lfs = []
                    orig_imgs = []
                    
                    psnrs_1 = []
                    ssims_1 = []
                    
                    curr_img = batch[id]['image'].to(self.device1)
                    prev_img = batch[max(0, id-1)]['image'].to(self.device1)
                    next_img = batch[min(leng, id+1)]['image'].to(self.device1)

                    prev_orig_image = denormalize(prev_img, self.device1)
                    curr_orig_image = denormalize(curr_img, self.device1)
                    next_orig_image = denormalize(next_img, self.device1)

                    gt_lf = batch[id]['lf'].to(self.device)
                    prev_gt_lf = batch[max(0, id-1)]['lf'].to(self.device)

                    dpt_disp = batch[id]['disp'].to(self.device1)
                    disp = -1 * (dpt_disp - zp) * md
                    img = torch.cat([prev_img, curr_img, next_img, disp], dim=1)
                
                    img = img.to(self.device1)
                    decomposition, depth_planes, state = self.lf_model(img, prev_state)
                    pred_lf = self.val_td_model(decomposition, depth_planes)
                    pred_lf = pred_lf.clip(0, 1)

                    curr_img = curr_img.to(self.device)
                    pred_lf = pred_lf.to(self.device)

                    lf_inp = torch.cat([pred_lf, curr_img.unsqueeze(1)], dim=1)
                    mask, corr_lf = self.ref_model(lf_inp)
                    ref_lf = mask*corr_lf + (1-mask)*pred_lf
                    ref_lf = ref_lf.clip(0, 1)
                    
                    temp_loss = self.compute_losses(gt_lf, prev_gt_lf, ref_lf)
                    temp_loss_avg.append(temp_loss)

                    string = 'Sample {0:2d} => Temp loss: {1:.6f}\n'.format(i, temp_loss)
                    f.write(string)
                    
                    vepoch.set_postfix(temp_loss=f"{temp_loss_avg.get_value():0.6f}({temp_loss:0.6f})",)

        avg_temp_loss = temp_loss_avg.get_value()
        string = 'Average Temp Loss: {0:.6f}\n'.format(avg_temp_loss)
        f.write(string)
        f.close()


    def main_worker(self):        
        ###############################################################################################
        test_loader = LFDataLoader(self.args, 'eval').data
        zp, md = self.zp, self.md
        self.test(test_loader, md, zp)



def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Testing script. Default values of all arguments are recommended for reproducibility', 
                                     fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('--results', default='results', type=str, help='directory to save results')
    parser.add_argument('-sn', '--save_numpy', default=False, action='store_true', help='whether to save np files')
    
    parser.add_argument('--gpu_1', default=1, type=int, help='which gpu to use')
    parser.add_argument('--gpu_2', default=0, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=1, type=int, help='number of workers for data loading')

    ######################################## Dataset parameters #######################################
    parser.add_argument('-d', '--dataset', default='Kalantari', type=str, help='Dataset to train on')

    parser.add_argument('--lf_path', default='/media/data/prasan/datasets/LF_video_datasets/', type=str, 
                        help='path to the data for online evaluation')
    parser.add_argument('--disp_path', default='/media/data/prasan/datasets/LF_video_datasets/DPT-depth', type=str, 
                        help='path to the groundtruth data for online evaluation')

    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')

    ############################################# I/0 parameters ######################################
    parser.add_argument('-h', '--height', type=int, help='input height', default=352)
    parser.add_argument('-w', '--width', type=int, help='input width', default=528)
    parser.add_argument('-md', '--max_displacement', default=1.2, type=float)
    parser.add_argument('-zp', '--zero_plane', default=0.3, type=float)
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true')

    ####################################### RAFT parameters ###########################################
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    ##################################### Learning parameters #########################################
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')

    ##################################### Tensor Display parameters #########################################
    parser.add_argument('--rank', default= 12, type=int, help='rank of the tensor display')
    parser.add_argument('--num_layers', default= 3, type=int, help='number of layers in the tensor display')
    parser.add_argument('--angular', default= 7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')

    args = parser.parse_args()
    args.filenames_file_eval = f'test_inputs/{args.dataset}/test_files.txt'

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    tester = Tester(args)
    tester.main_worker()