import time
import torch
from torchvision import utils, transforms
from imgseq import ImageSequence
from torch.autograd import Variable
from torch import optim
from mefssim import MEF_MSSSIM, MEFSSIM, mef_ssim
from batch_transformers import BatchRandomResolution, BatchRGBToYCbCr, BatchToTensor, YCbCrToRGB
from PIL import Image
import numpy as np


EPS = 1e-8
img_path = './mef_dataset_512/5/011/'


transform = transforms.Compose([
    BatchRandomResolution(512, interpolation=2),
    BatchToTensor(),
    BatchRGBToYCbCr(),
])

dataset = ImageSequence(img_path, transform=transform)
# x = BatchToTensor()([Image.open('./mertens07_tower.png')])
# x = BatchRGBToYCbCr()(x)
# x = x[0][0,:,:].unsqueeze(0).unsqueeze(1)
x = torch.mean(dataset[0][:, 0, :, :], dim=0).unsqueeze(0).unsqueeze(1)
# x = torch.zeros_like(dataset[0][0, 0, :, :]).unsqueeze(0).unsqueeze(1)
Seq = Variable(dataset[0], requires_grad=False)

mefssim_loss = MEF_MSSSIM(is_lum=True)

if torch.cuda.is_available():
    x = x.cuda()
    Seq = Seq.cuda()
    mefssim_loss = mefssim_loss.cuda()


X = x.detach().requires_grad_()
Y = Seq[:, 0, :, :].unsqueeze(1)
Cb = Seq[:, 1, :, :].unsqueeze(1)
Cr = Seq[:, 2, :, :].unsqueeze(1)

optimizer = optim.Adam([X], lr=0.01)

Wb = (torch.abs(Cb - 0.5) + EPS) / torch.sum(torch.abs(Cb - 0.5) + EPS, dim=0)
Wr = (torch.abs(Cr - 0.5) + EPS) / torch.sum(torch.abs(Cr - 0.5) + EPS, dim=0)
Cb_f = torch.sum(Wb * Cb, dim=0, keepdim=True)
Cr_f = torch.sum(Wr * Cr, dim=0, keepdim=True)

iter = 0
while iter < 500:
    start_time = time.time()

    optimizer.zero_grad()
    q_out = -mefssim_loss(X, Y)
    q = - q_out.item()
    q_out.backward()
    optimizer.step()

    current_time = time.time()
    duration = current_time - start_time
    format_str = '(S:%d) MEF-SSIM = %.4f (%.3f sec/step)'
    print(format_str % (iter, q, duration))
    X.data.clamp_(0, 1)
    X_color = YCbCrToRGB()(torch.cat((X, Cb_f, Cr_f), dim=1))
    utils.save_image(X_color.data, "./img_opt_ycbcr/%d.png" % (iter))
    iter += 1


