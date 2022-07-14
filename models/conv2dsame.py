import torch
import torch.nn as nn
import torch.nn.functional as F

def _calc_same_pad(i, k, s, d):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def conv2d_same(x, weight, bias = None, stride = (1, 1), padding = (0, 0), dilation = (1, 1), groups = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
