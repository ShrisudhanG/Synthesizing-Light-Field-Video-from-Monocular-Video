import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2dsame import Conv2dSame


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return Conv2dSame(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, 
                      dilation=dilation)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None, 
                groups= 1, base_width = 64, dilation = 1, norm_layer = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(32, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(32, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(32, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()#DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class UpSampleConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSampleConcat, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_feat),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_feat),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class RefinementBlock(nn.Module):
    def __init__(self, patch_size=2, groups=1, width_per_group=64, norm_layer=None):
        super(RefinementBlock, self).__init__()

        self.patch_size = patch_size
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        
        self.inplanes = 64
        self.dilation = 1
        
        self.conv_in = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2), 
                                     nn.BatchNorm2d(self.inplanes),
                                     nn.LeakyReLU(inplace=True))
        
        self.down1 = self._make_layer(Bottleneck, 64, 3, stride=2, dilate=False)
        self.down2 = self._make_layer(Bottleneck, 128, 3, stride=2, dilate=False)
        self.down3 = self._make_layer(Bottleneck, 256, 3, stride=2, dilate=False)
        self.down4 = self._make_layer(Bottleneck, 512, 3, stride=2, dilate=False)
        expansion = Bottleneck.expansion

        self.proj1 = nn.Linear(512*expansion*patch_size**2, 768)
        self.attn1 = Attention(768, 12)
        self.attn2 = Attention(768, 12)
        self.proj2 = nn.Linear(768, 512*expansion*patch_size**2)

        self.outplanes = 512*expansion
        self.up1 = UpSampleConcat(self.outplanes + 256*expansion, self.outplanes//2)
        self.up2 = UpSampleConcat(self.outplanes//2 + 128*expansion, self.outplanes//4)
        self.up3 = UpSampleConcat(self.outplanes//4 + 64*expansion, self.outplanes//8)
        self.up4 = UpSampleConcat(self.outplanes//8 + 64, self.outplanes//8)

        self.conv_out = nn.Sequential(nn.Conv2d(self.outplanes//8, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1))
        self.sigmoid = nn.Sigmoid()


    def get_patches(self, features, patch_size):
        N, V, C, H, W = features.shape
        #print(H, W)
        if H % patch_size != 0:
            raise ValueError(f'height({H}) should be divisible by patch_size({patch_size})')
        if W % patch_size != 0:
            raise ValueError(f'width({W}) should be divisible by patch_size({patch_size})')
        h, w = patch_size, patch_size
        h_patches = H//h
        w_patches = W//w
        num_patches = (h_patches, w_patches)

        '''
        patch_features = torch.zeros((N, h_patches*w_patches, V, C*patch_size[0]*patch_size[1]))
        for i in range(h_patches):
            for j in range(w_patches):
                patch_feat = features[:, :, :, i*patch_size[0]:(i+1)*patch_size[0],
                                      j*patch_size[1]:(j+1)*patch_size[1]]
                patch_feat = patch_feat.resize(N, V, -1)
                patch_features[:, i*w_patches+j, :, :] = patch_feat
        '''
        patch_features = features.reshape(N, V, C, h_patches, w_patches, h, w)
        patch_features = patch_features.permute(0, 1, 2, 3, 5, 4, 6)
        patch_features = patch_features.reshape(N, V, C, h_patches*w_patches, h, w)
        patch_features = patch_features.permute(0, 3, 1, 2, 4, 5)
        patch_features = patch_features.reshape(N, h_patches*w_patches, V, C*h*w)

        return patch_features, num_patches

    
    def assemble_patches(self, patch_features, num_patches, patch_size):
        N, _, V, F = patch_features.shape
        h_patches, w_patches = num_patches
        if _ != h_patches*w_patches:
            raise ValueError(f'Expected patch_features to have {h_patches*w_patches} channels but received {_} channels')
        h, w = patch_size, patch_size
        C = (F//h)//w
        H = int(h_patches * patch_size)
        W = int(w_patches * patch_size)

        '''
        features = torch.zeros((N, V, C, H, W))
        for i in range(h_patches):
            for j in range(w_patches):
                patch_feat = patch_features[:, i*w_patches+j, :, :]
                patch_feat = patch_feat.reshape(N, V, C, patch_size[0], patch_size[1])
                features[:, :, :, i*patch_size[0]:(i+1)*patch_size[0], 
                         j*patch_size[1]:(j+1)*patch_size[1]] = patch_feat
        '''

        recon_features = patch_features.reshape(N, h_patches*w_patches, V, C, h, w)
        recon_features = recon_features.permute(0, 2, 3, 1, 4, 5)
        recon_features = recon_features.reshape(N, V, C, h_patches, h, w_patches, w)
        recon_features = recon_features.permute(0, 1, 2, 3, 5, 4, 6)
        recon_features = recon_features.reshape(N, V, C, H, W)

        return recon_features


    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(32, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, 
                                base_width=self.base_width, dilation=self.dilation, 
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x_batch):
        N, V, Ci, Hi, Wi = x_batch.shape
        x = x_batch.reshape(N*V, Ci, Hi, Wi)
        patch_size = self.patch_size
            
        feat1 = self.conv_in(x)
        feat2 = self.down1(feat1)
        feat3 = self.down2(feat2)
        feat4 = self.down3(feat3)
        feat5 = self.down4(feat4)

        _, C1, H1, W1 = feat5.shape
        feat = feat5.reshape(N, V, C1, H1, W1)
        patch_feat, num_patches = self.get_patches(feat, patch_size)

        total_patches = num_patches[0]*num_patches[1]
        patch_feat = patch_feat.reshape(N*total_patches*V, C1*patch_size**2)
        patch_feat = self.proj1(patch_feat) # 2d input required. check resizing op.
        patch_feat = patch_feat.reshape(N*total_patches, V, -1)

        attn1_out = self.attn1(patch_feat)
        attn2_out = self.attn2(attn1_out)
        
        attn2_out = attn2_out.reshape(N*total_patches*V, -1)
        attn_patch_out = self.proj2(attn2_out) # 2d input required. check resizing op.
        attn_patch_out = attn_patch_out.reshape(N, total_patches, V, C1*patch_size**2)

        attn_feat = self.assemble_patches(attn_patch_out, num_patches, patch_size)
        attn_feat = attn_feat[:, :-1, ...]
        _, V, C2, H2, W2 = attn_feat.shape
        attn_feat = attn_feat.reshape(N*V, C2, H2, W2)

        feat6 = self.up1(attn_feat, feat4[:-1, ...])
        feat7 = self.up2(feat6, feat3[:-1, ...])
        feat8 = self.up3(feat7, feat2[:-1, ...])
        feat9 = self.up4(feat8, feat1[:-1, ...])
        feat9 = F.interpolate(feat9, (Hi, Wi))
        feat10 = self.conv_out(feat9)

        _, Co, Ho, Wo = feat10.shape
        feat10 = feat10.reshape(N, V, Co, Ho, Wo)
        mask, output = feat10[:, :, 0, :, :], feat10[:, :, 1:, :, :]
        mask = self.sigmoid(mask).unsqueeze(2)

        return mask, output


if __name__ == '__main__':
    from tqdm import tqdm
    import torch.optim as optim
    #import lpips
    from lpips_pytorch import LPIPS

    device = torch.device('cuda:3')
    model = RefinementBlock(patch_size=1).to(device)
    loss_fn = nn.L1Loss()
    #loss_fn = LPIPS(net_type='vgg').to(device)
    params = model.parameters()
    optimizer = optim.AdamW(params, weight_decay=0.1, lr=1e-4)
    model.train()

    for i in tqdm(range(3000)):
        optimizer.zero_grad()
        input = torch.rand([1, 49, 3, 180, 270]).to(device)
        gt = input + 0.1*torch.rand(input.shape).to(device)
        mask, output = model(input)
        output = mask*output + (1-mask)*input
        loss = loss_fn(output[0], gt[0])
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(loss.item())

    print(mask.shape, output.shape)