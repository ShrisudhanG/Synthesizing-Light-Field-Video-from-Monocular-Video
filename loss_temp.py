import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


class L1Loss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(L1Loss, self).__init__()
        self.name = 'L1 Loss'

    def forward(self, inp, tar):
        diff = (inp - tar)**2
        loss = diff.mean()
        return loss


class TemporalConsistency(nn.Module):
    """docstring for temporal_criterion"""
    def __init__(self, args, device):
        super(TemporalConsistency, self).__init__()
        self.name = 'Temporal Consistency Loss'
        self.args = args
        self.device = device
        self.angular = args.angular
        self.diff_loss = L1Loss()
        self.flownet = nn.DataParallel(RAFT(args), device_ids=[args.gpu_1])
        self.flownet.load_state_dict(torch.load('raft-things.pth', map_location='cpu'))
        self.flownet = self.flownet.to(self.device)


    def init_default_flow(self, img):
        _, _, _, h, w = img.shape
        x = np.linspace(-1., 1., w)
        y = np.linspace(-1., 1., h)
        xv,yv = np.meshgrid(x, y)
        default_flow = np.zeros((h, w, 2))
        default_flow[..., 0] = xv
        default_flow[..., 1] = yv
        default_flow = default_flow[None, ...]
        default_flow = torch.FloatTensor(default_flow)

        self.default_flow = default_flow.to(self.device)
        self.H = h
        self.W = w

    
    def get_flow(self, curr_img, prev_img):
        self.flownet.eval()
        N, V, C, H, W = curr_img.shape
        curr_img = curr_img.reshape(N*V, C, H, W)
        prev_img = prev_img.reshape(N*V, C, H, W)
        with torch.no_grad():
            padder = InputPadder(curr_img.shape)
            curr_img, prev_img = padder.pad(curr_img, prev_img)

            _, flow = self.flownet(prev_img, curr_img, iters=20, test_mode=True)
        return flow
    

    def forward(self, curr_lf, prev_lf, pred_lf):
        self.init_default_flow(curr_lf)
        N, V, C, H, W = pred_lf.shape
        pred_lf = pred_lf.reshape(N*V, C, H, W)

        flow = self.get_flow(curr_lf, prev_lf)
        flow = flow.permute(0,2,3,1).contiguous()
        flow[...,0] = 2 * flow[...,0] / self.W
        flow[...,1] = 2 * flow[...,1] / self.H
        flow = flow + self.default_flow

        warped_pred_lf = F.grid_sample(pred_lf, flow, mode='bilinear', align_corners=True)
        warped_pred_lf = warped_pred_lf.reshape(N, V, C, H, W)
        
        temp_loss = self.diff_loss(warped_pred_lf, prev_lf)
        
        return temp_loss