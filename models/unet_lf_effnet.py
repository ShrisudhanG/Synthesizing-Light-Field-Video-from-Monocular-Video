import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT
from .convlstm import ConvLSTM

# oc*self.rank*self.n_layers

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048, layers=3, rank=12):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.n_layers, self.rank = layers, rank
        self.conv2 = nn.Conv2d(bottleneck_features, features//2, kernel_size=1, stride=1, padding=1)
        
        self.conv_lstm = ConvLSTM(input_size=features//2, hidden_size=features//2, kernel_size=3)

        self.up1 = UpSampleBN(skip_input=features // 2 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        self.up_x = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, features, prev_state):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        state = self.conv_lstm(x_d0, prev_state)
        x_d0 = state[1]
        #state = None

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        x_d5 = self.up_x(x_d4)
        out = self.conv3(x_d5)
        out = self.relu(out)
        return out, state


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UnetLF(nn.Module):
    def __init__(self, backend, td_chans=100, layers=None, rank=None):
        super(UnetLF, self).__init__()
        self.layers = layers
        self.rank = rank
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(num_classes=td_chans, layers=layers, rank=rank)
        self.adaptive_bins_layer = mViT(10, patch_size=16, dim_out=layers, embedding_dim=128)


    def forward(self, x, prev_state, **kwargs):
        out = self.encoder(x)
        unet_out, state = self.decoder(out, prev_state, **kwargs)

        depth_planes = self.adaptive_bins_layer(x)
        
        N, C, h, w = unet_out.size()
        unet_out = unet_out.view(N, self.layers, self.rank, 3, h, w)

        return unet_out, depth_planes, state
        

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        #print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        #print('Done.')

        # Remove last layer
        #print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        #print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, **kwargs)
        #print('Done.')
        return m


if __name__ == '__main__':
    model = UnetLF.build()
    x = torch.rand(2, 3, 480, 640)
    pred = model(x)
    print(pred.shape)