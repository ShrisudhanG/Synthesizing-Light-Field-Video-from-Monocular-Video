import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def get_default_flow(b, h, w):
    x = np.linspace(-1., 1., w)
    y = np.linspace(-1., 1., h)
    xv, yv = np.meshgrid(x, y)
    default_flow = np.zeros((h, w, 2))
    default_flow[..., 0] = xv
    default_flow[..., 1] = yv
    return torch.FloatTensor(default_flow)


class multilayer(nn.Module):
    def __init__(self, height, width, reduce_mean=True, args=None, device=None):
        super(multilayer,self).__init__()
        # define the filters
        h = height
        w = width
        n_layers = args.num_layers
        angular = args.angular
        self.reduce_mean = reduce_mean
        self.factor = args.td_factor
        self.device = device
        factor_h = self.factor*(-1.0/h)
        factor_w = self.factor*(-1.0/w)
        self.ang = angular
        self.default_flow = get_default_flow(1, h, w).to(self.device)
        layer = np.zeros((angular**2, h, w, 2))
        a = 0
        for k in range(-angular//2+1, angular//2+1):
            for l in range(-angular//2+1, angular//2+1):
                layer[a, :, :, 0] = factor_w * l
                layer[a, :, :, 1] = factor_h * k
                a += 1
        self.layer = torch.FloatTensor(layer).to(self.device)
        #print(self.layer.shape, self.default_flow.shape)
        #print(self.filters.size())

    def cust_expand(self, l):
        N,_,h,w = l.size()
        # l = F.relu(l)
        l = l.expand(N, self.a**2, h, w)
        l = l.unsqueeze(2)
        return l

    def forward(self, low_rank, planes):
        N, n_layers = planes.shape
        filter = self.layer.unsqueeze(0).unsqueeze(0)
        filter = filter.repeat(N, 1, 1, 1, 1, 1)
        filters = []
        for i in range(n_layers):
            d = planes[:, i]
            d = d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            filters.append(filter*d)
        filters = torch.cat(filters, dim=1)
        filters += self.default_flow[None, None, None, ...]
        # filter shape = (N, layers, angular**2, h, w, 2)
        
        N, num_layers, rank, c, h, w = low_rank.size() # c is the RGB channel
        lf = []
        #filters = self.filters.unsqueeze(0).expand(N,num_layers,self.ang**2,h,w,2)
        # layers is of shape [n_layers,T,h,w]
        for a in range(self.ang**2):
            layers_shift_prod = torch.ones(N, rank*c, h, w).to(self.device)
            for l in range(num_layers):
                layer = low_rank[:, l, ...].view(N, rank*c, h, w)
                layers_shift = F.grid_sample(layer, filters[:, l, a, ...], 
                padding_mode='border', mode='bilinear', align_corners=True)
                layers_shift_prod = layers_shift_prod*layers_shift
                # lf_append = torch.prod(layers_shift,0,keepdim=True)
            layers_shift_prod = layers_shift_prod.view(N, rank, c, h, w)
            if self.reduce_mean:
                lf.append(layers_shift_prod.mean(1, keepdim=True))
            else:
                lf.append(layers_shift_prod)
        if self.reduce_mean:
            lf = torch.cat(lf, 1)
            return lf
        else:
            lf = torch.stack(lf, 2)
            return lf




# with angular backlight
class tensor_display(nn.Module):
    def __init__(self,angular, layers, h, w, args=None):
        super(tensor_display,self).__init__()
        self.u = int(angular)

    def kronecker_product(self,back,front):
        prod = torch.einsum('nchwij,nchwkl->nhwijkl', back, front)
        return prod

    def forward(self,layers):
        # so the num_layers will be 2 in this case
        # a back layer and a front layer
        # the height and width of layers should be a multiple of self.u
        back = layers[:, 0, ...]
        front = layers[:, 1, ...]
        assert back.size(-1)%self.u == 0
        assert back.size(-2)%self.u == 0
        N, c, h, w = back.size()
        back = back.unfold(2, self.u, self.u).unfold(3, self.u, self.u)
        front = front.unfold(2, self.u, self.u).unfold(3, self.u, self.u)
        lf_video = self.kronecker_product(back, front)
        t, ha, wa, a, _, _, _ = lf_video.size()
        lf_video = lf_video.reshape(t, -1, a*a*a*a)
        lf_video = lf_video.permute(0, 2, 1)
        lf_video = torch.nn.functional.fold(lf_video, (ha*a,wa*a), a, stride=a)
        #lf_video = lf_video.permute(0, 5, 6, 1, 3, 2, 4)
        lf_video = lf_video.reshape(t, a*a, ha*a, wa*a)
        # lf = lf.permute(0, 3, 4, 1, 5, 2, 6)
        # lf = lf.reshape(N, self.u*self.u, h, w)
        return lf_video/c