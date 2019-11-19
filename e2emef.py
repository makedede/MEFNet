import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter


EPS = 1e-8


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('InstanceNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)


def build_lr_net(norm=AdaptiveNorm, layer=5, width=24):#lr = low resolution
    layers = [
        nn.Conv2d(1, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(width),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(width,  1, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init_identity)

    return net


class E2EMEF(nn.Module):
    # end-to-end mef model
    def __init__(self, radius=1, eps=1e-4, is_guided=True):
        super(E2EMEF, self).__init__()
        self.lr = build_lr_net()
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        w_lr = self.lr(x_lr)

        if self.is_guided:
            w_hr = self.gf(x_lr, w_lr, x_hr)
        else:
            w_hr = F.upsample(w_lr, x_hr.size()[2:], mode='bilinear')

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + EPS) / torch.sum((w_hr + EPS), dim=0)

        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)

        return o_hr, w_hr

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))
