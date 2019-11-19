import time
import torch
from torchvision import utils, transforms
from imgseq import ImageSequence
from torch.autograd import Variable
from torch import optim
from mefssim import MEF_MSSSIM, MEFSSIM, mef_ssim
from batch_transformers import BatchRandomResolution, BatchRGBToYCbCr, BatchToTensor, YCbCrToRGB

img_path = './House_Tom Mertens09/'
# img_path = './dataset/hr/a/'

transform = transforms.Compose([
    BatchRandomResolution(1792, interpolation=2),
    BatchToTensor(),
])

dataset = ImageSequence(img_path, transform=transform)

x = torch.mean(dataset[0], dim=0, keepdim=True)
Seq = Variable(dataset[0], requires_grad=False)

mefssim_loss = MEF_MSSSIM(is_lum=True)

if torch.cuda.is_available():
    x = x.cuda()
    Seq = Seq.cuda()
    mefssim_loss = mefssim_loss.cuda()


X = x.detach().requires_grad_()

optimizer = optim.Adam([X], lr=0.01)

iter = 0
while iter < 500:
    start_time = time.time()

    optimizer.zero_grad()
    q_out = -mefssim_loss(X, Seq)
    q = - q_out.item()
    q_out.backward()
    optimizer.step()

    current_time = time.time()
    duration = current_time - start_time
    format_str = '(S:%d) MEF-SSIM = %.4f (%.3f sec/step)'
    print(format_str % (iter, q, duration))
    X.data.clamp_(0, 1)
    # utils.save_image(X.data, "./results/%d.png" % (iter))
    iter += 1

# grid = utils.make_grid(X.data)
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.show()

