import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

import softsplat
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


class L1Loss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(L1Loss, self).__init__()
        self.name = 'L1 Loss'
        

    def forward(self, inp, tar, weight_mask=None, valid_mask=None):
        mask = (tar > 0).to(bool) * (inp > 0).to(bool)
        if valid_mask is not None:
            mask = mask * valid_mask.to(bool)
        minp = inp[mask]
        mtar = tar[mask]
        diff = torch.abs(minp - mtar)
        if weight_mask is not None:
            mweight = weight_mask[mask]
            diff = diff * mweight
        loss = diff.mean()
        return 10 * loss



class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super(BinsChamferLoss, self).__init__()
        self.name = "ChamferLoss"


    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss



class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2


    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return 10 * torch.mean(torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1))


class SmoothLoss(nn.Module):
    def __init__(self, args, device):
        super(SmoothLoss, self).__init__()
        self.name = 'Smoothness Loss'
        self.args = args
        gradx = torch.FloatTensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).to(device)
        grady = torch.FloatTensor([[-1, -2, -1],
                                   [0,   0,  2],
                                   [1,   0,  1]]).to(device)
        self.disp_gradx = gradx.unsqueeze(0).unsqueeze(0)
        self.disp_grady = grady.unsqueeze(0).unsqueeze(0)
        self.img_gradx = self.disp_gradx.repeat(1, 3, 1, 1)
        self.img_grady = self.disp_grady.repeat(1, 3, 1, 1)


    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        grad_disp_x = torch.abs(F.conv2d(disp, self.disp_gradx, padding=1, stride=1))
        grad_disp_y = torch.abs(F.conv2d(disp, self.disp_grady, padding=1, stride=1))

        grad_img_x = torch.abs(torch.mean(F.conv2d(img, self.img_gradx, padding=1, stride=1), dim=1, keepdim=True))
        grad_img_y = torch.abs(torch.mean(F.conv2d(img, self.img_grady, padding=1, stride=1), dim=1, keepdim=True))

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        loss_x = 10 * (torch.sqrt(torch.var(grad_disp_x) + 0.15 * torch.pow(torch.mean(grad_disp_x), 2)))
        loss_y = 10 * (torch.sqrt(torch.var(grad_disp_y) + 0.15 * torch.pow(torch.mean(grad_disp_y), 2)))

        return loss_x + loss_y


    def forward(self, decomposition, disp):
        N, layers, rank, C, H, W = decomposition.shape
        disp = disp.unsqueeze(dim=1)
        disp = disp.repeat(1, layers*rank, 1, 1, 1)
        disp = disp.reshape(-1, 1, H, W)
        decomposition = decomposition.reshape(-1, C, H, W)
        loss = self.get_smooth_loss(disp, decomposition)
        return loss


class PhotometricConsistency(nn.Module):
    def __init__(self, args, device):
        super(PhotometricConsistency, self).__init__()
        self.name = 'Photometric Consistency Loss'
        self.args = args
        self.angular = args.angular
        self.device = device
        self.diff_loss = nn.L1Loss()
        self.is_ssim = args.ssim
        self.w_ssim = args.w_ssim
        self.ssim_loss = SSIMLoss().to(device)


    def forward(self, img, pred_lf):
        center_pred = pred_lf[:, int(self.angular**2//2), ...]
        photo_loss = self.diff_loss(center_pred, img)
        if self.is_ssim:
            photo_loss +=  self.w_ssim * self.ssim_loss(center_pred, img)
        return photo_loss



class GeometricConsistency(nn.Module):
    def __init__(self, args, device):
        super(GeometricConsistency, self).__init__()
        self.name = 'Geometric Consistency Loss'
        self.args = args
        self.angular = args.angular
        self.device = device
        #self.max_disp = args.max_displacement
        self.diff_loss = L1Loss()
        self.use_mask = args.edge_weight_mask
        self.is_ssim = args.ssim
        self.w_ssim = args.w_ssim
        self.ssim_loss = SSIMLoss().to(device)
        
        x_factor = np.arange(-1 * self.angular//2 + 1, self.angular//2 + 1)
        y_factor = np.arange(-1 * self.angular//2 + 1, self.angular//2 + 1)
        factor = np.stack([np.meshgrid(x_factor, y_factor)], 2)
        factor = factor.squeeze().transpose(1, 2, 0).reshape(1, self.angular**2, 2, 1, 1)
        self.factor = torch.FloatTensor(factor).to(self.device)

        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])
        sobel_y = sobel_x.T
        self.gradx = torch.FloatTensor(sobel_x[None, None, ...]).to(device)
        self.grady = torch.FloatTensor(sobel_y[None, None, ...]).to(device)


    def gradxy(self, tensor):
        tensor = tensor.mean(dim=1, keepdim=True)
        gradx = F.conv2d(tensor, self.gradx, padding=1, stride=1)
        grady = F.conv2d(tensor, self.grady, padding=1, stride=1)
        grad = gradx.abs() + grady.abs()
        return grad


    def forward_warp(self, img, depth):
        curr_img = img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1)
        #depth = self.max_disp * depth
        N, _, H, W = depth.shape
        _, C, _, _ = img.shape

        depth = depth.unsqueeze(1)
        depth = depth.repeat(1, self.angular**2, 2, 1, 1)
        depth = depth*self.factor

        depth = depth.reshape(N * self.angular**2, 2, H, W)
        curr_img = curr_img.reshape(N * self.angular**2, C, H, W)
        
        warped_lf = softsplat.FunctionSoftsplat(tenInput=curr_img, tenFlow=depth, tenMetric=None, strType='average').to(self.device)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        return warped_lf

    
    def init_coord(self, img):
        _, _, h, w = img.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        coord = np.zeros((2, h, w))
        coord[0, ...] = xv
        coord[1, ...] = yv
        coord = np.tile(coord, [1, self.angular**2, 1, 1, 1])
        self.coord = torch.FloatTensor(coord).to(self.device)


    def forward(self, img, depth, pred_lf):
        self.init_coord(img)
        
        N, _, H, W = depth.shape
        _, V, C, _, _ = pred_lf.shape

        depth = depth.unsqueeze(1)
        depth = depth.repeat(1, self.angular**2, 2, 1, 1)
        depth = depth * self.factor
        depth[:, :, 0, :, :] /= W/2
        depth[:, :, 1, :, :] /= H/2

        warp_coord = self.coord + depth
        warp_coord = warp_coord.reshape(N * self.angular**2, 2, H, W).permute(0, 2, 3, 1)
        pred_lf = pred_lf.reshape(N * self.angular**2, C, H, W)

        warped_lf = F.grid_sample(pred_lf, warp_coord, padding_mode='border', mode='bilinear', align_corners=True)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        if self.use_mask:
            weight_mask = self.gradxy(img).unsqueeze(1).repeat(1, self.angular**2, 3, 1, 1)
            geo_loss = self.diff_loss(warped_lf, img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1), weight_mask)
        else:
            geo_loss = self.diff_loss(warped_lf, img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1))
        
        warped_lf = warped_lf.reshape(N*self.angular**2, C, H, W)
        img = img.repeat(self.angular**2, 1, 1, 1)
        if self.is_ssim:
            geo_loss +=  self.w_ssim * self.ssim_loss(warped_lf, img)

        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        return geo_loss, warped_lf


class TemporalConsistency(nn.Module):
    """docstring for temporal_criterion"""
    def __init__(self, args, device):
        super(TemporalConsistency, self).__init__()
        self.name = 'Temporal Consistency Loss'
        self.args = args
        self.device = device
        self.angular = args.angular
        self.diff_loss = L1Loss()
        self.use_mask = args.edge_weight_mask
        self.is_ssim = args.ssim
        self.w_ssim = args.w_ssim
        self.ssim_loss = SSIMLoss().to(device)
        self.flownet = nn.DataParallel(RAFT(args), device_ids=[args.gpu])
        self.flownet.load_state_dict(torch.load('raft-things.pth', map_location='cpu'))
        self.flownet = self.flownet.to(self.device)

        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])
        sobel_y = sobel_x.T
        self.gradx = torch.FloatTensor(sobel_x[None, None, ...]).to(device)
        self.grady = torch.FloatTensor(sobel_y[None, None, ...]).to(device)


    def gradxy(self, tensor):
        tensor = tensor.mean(dim=1, keepdim=True)
        gradx = F.conv2d(tensor, self.gradx, padding=1, stride=1)
        grady = F.conv2d(tensor, self.grady, padding=1, stride=1)
        grad = gradx.abs() + grady.abs()
        return grad


    def init_default_flow(self, img):
        _, _, h, w = img.shape
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
        N, C, H, W = curr_img.shape
        with torch.no_grad():
            padder = InputPadder(curr_img.shape)
            curr_img, prev_img = padder.pad(curr_img, prev_img)

            _, flow = self.flownet(prev_img, curr_img, iters=20, test_mode=True)
        return flow
    

    def forward(self, curr_img, prev_img, warped_lf):
        self.init_default_flow(curr_img)
        N, V, C, H, W = warped_lf.shape
        warped_lf = warped_lf.reshape(N*V, C, H, W)

        flow = self.get_flow(curr_img, prev_img)
        flow = flow.repeat(V, 1, 1, 1)

        flow = flow.permute(0,2,3,1).contiguous()
        flow[...,0] = 2 * flow[...,0] / self.W
        flow[...,1] = 2 * flow[...,1] / self.H
        flow = flow + self.default_flow
        warped_prev_lf = F.grid_sample(warped_lf, flow, mode='bilinear', align_corners=True)
        warped_prev_lf = warped_prev_lf.reshape(N, V, C, H, W)
        
        if self.use_mask:
            weight_mask = self.gradxy(prev_img).unsqueeze(1).repeat(1, self.angular**2, 3, 1, 1)
            temp_loss = self.diff_loss(warped_prev_lf, prev_img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1), weight_mask)
        else:
            temp_loss = self.diff_loss(warped_prev_lf, prev_img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1))
        
        warped_prev_lf = warped_prev_lf.reshape(N*self.angular**2, C, H, W)
        prev_img = prev_img.repeat(self.angular**2, 1, 1, 1)
        if self.is_ssim:
            temp_loss +=  self.w_ssim * self.ssim_loss(warped_prev_lf, prev_img)

        warped_prev_lf = warped_prev_lf.reshape(N, self.angular**2, C, H, W)
        return temp_loss, warped_prev_lf



def backwarp(tenInput, tenFlow):
    backwarp_tenGrid = {}
    device = tenInput.device
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
                                tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
                                tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).to(device)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(tenInput, (backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), 
                                           mode='bilinear', padding_mode='zeros', align_corners=False)


class OcclusionLoss(nn.Module):
    def __init__(self, args, device, ratio):
        super(OcclusionLoss, self).__init__()
        self.name = 'Occlusion Handling Loss'
        self.args = args
        self.angular = args.angular
        self.device = device
        self.ratio = ratio
        self.diff_loss = L1Loss()
        self.flownet = nn.DataParallel(RAFT(args), device_ids=[args.gpu])
        self.flownet.load_state_dict(torch.load('raft-things.pth'))
        self.flownet = self.flownet.to(self.device)
        
        x_factor = np.arange(-1 * self.angular//2 + 1, self.angular//2 + 1)
        y_factor = np.arange(-1 * self.angular//2 + 1, self.angular//2 + 1)
        factor = np.stack([np.meshgrid(x_factor, y_factor)],2)
        factor = factor.squeeze().transpose(1, 2, 0).reshape(1, self.angular**2, 2, 1, 1)
        self.factor = -1 * torch.FloatTensor(factor).to(self.device)

    
    def init_coord(self, img):
        _, _, h, w = img.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        coord = np.zeros((2, h, w))
        coord[0, ...] = xv
        coord[1, ...] = yv
        coord = np.tile(coord, [1, self.angular**2, 1, 1, 1])
        self.coord = torch.FloatTensor(coord).to(self.device)


    def get_mask(self, img, depth, inds):
        curr_img = img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1)
        #depth = self.max_disp * depth
        N, _, H, W = depth.shape
        _, C, _, _ = img.shape

        depth = depth.unsqueeze(1)
        depth = depth.repeat(1, self.angular**2, 2, 1, 1)
        depth = depth*self.factor

        depth = depth.reshape(N * self.angular**2, 2, H, W)
        curr_img = curr_img.reshape(N * self.angular**2, C, H, W)

        depth[:, 0, ...] = -1 * depth[:, 0, ...]
        depth[:, 1, ...] = -1 * depth[:, 1, ...]
        #warped_lf = softsplat.FunctionSoftsplat(tenInput=curr_img, tenFlow=depth, tenMetric=None, strType='average').to(self.device)
        warped_lf = softsplat.softsplat(tenIn=curr_img, tenFlow=depth, tenMetric=None, strMode='avg').to(self.device)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        mask = (warped_lf.max(dim=2, keepdim=True).values == 0).repeat(1, 1, 3, 1, 1).to(bool)
        
        return warped_lf, mask#, warped_lf_1


    def get_flow(self, img, lf):
        self.flownet.eval()
        N, C, H, W = img.shape
        with torch.no_grad():
            padder = InputPadder(img.shape)
            img, lf = padder.pad(img, lf)

            _, flow = self.flownet(img, lf, iters=20, test_mode=True)
        return flow


    def get_loss(self, warped_lfs, mask, pred_lf):
        losses = []
        new_mask = mask
        for frame_id in warped_lfs.keys():
            warped_lf = warped_lfs[frame_id]
            new_mask = new_mask * (warped_lf>0).to(bool)

        for frame_id in warped_lfs.keys():
            warped_lf = warped_lfs[frame_id]
            lf = pred_lf[new_mask]
            warped_lf = warped_lf[new_mask]
            loss = 10 * torch.abs(lf - warped_lf)
            losses.append(loss.unsqueeze(0))

        if new_mask.sum() == 0:
            return 0.
            #return Variable(torch.tensor([0.]), requires_grad=True).to(self.device)

        if self.args.loss_type == 'mean':
            loss = torch.cat(losses, dim=0)
            loss = loss.mean()
        elif self.args.loss_type == 'min':
            loss = torch.cat(losses, dim=0).min(dim=0).values
            loss = loss.mean()
        return loss


    def forward(self, imgs, depth, pred_lf):
        curr_img = imgs[0]
        self.init_coord(curr_img)
        N, C, H, W = curr_img.shape

        inds = np.random.random_integers(size=int(self.ratio*N*self.angular**2), 
                                         high=(N*self.angular**2)-1, low=0)
        inds = list(inds)
        #inds = list(range(N*self.angular**2))
        #print(len(inds))

        forward_warp_lf, mask = self.get_mask(curr_img, depth, inds)
        
        warped_imgs = {}
        warped_lfs = {}
        masked_lfs = {}

        forward_warp_lf = forward_warp_lf.reshape(N * self.angular**2, C, H, W)
        forward_warp_lf = forward_warp_lf[inds, ...] 
        mask = mask.reshape(N * self.angular**2, C, H, W)
        mask = mask[inds, ...]

        for frame_id in [-1, 1]:
            image = imgs[frame_id].unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1)
            image = image.reshape(N * self.angular**2, C, H, W)
            image = image[inds, ...]

            flow = self.get_flow(image, forward_warp_lf)

            metric = F.l1_loss(image, target=backwarp(tenInput=forward_warp_lf, tenFlow=flow), 
                               reduction='none').mean(1, True)
            #warped_lf = softsplat.FunctionSoftsplat(tenInput=image, tenFlow=flow, 
            #                                        tenMetric=-20.0*metric, strType='softmax').to(self.device)
            warped_lf = softsplat.softsplat(tenIn=image, tenFlow=flow, tenMetric=-20.0*metric, 
                                            strMode='soft').to(self.device)

            warped_lfs[frame_id] = warped_lf.clamp(0, 1)
            masked_lfs[frame_id] = (warped_lf * mask).clamp(0, 1)

        pred_lf = pred_lf.reshape(N * self.angular**2, C, H, W)
        pred_lf = pred_lf[inds, ...]
        occ_loss = self.get_loss(warped_lfs, mask, pred_lf)
        return occ_loss, [forward_warp_lf, warped_lfs, mask]