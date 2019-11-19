import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, utils
from e2emef import E2EMEF
from mefssim import MEF_MSSSIM
from ImageDataset import ImageSeqDataset
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution

EPS = 1e-8


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.train_hr_transform = transforms.Compose([
            BatchRandomResolution(config.high_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])

        self.train_lr_transform = transforms.Compose([
            BatchRandomResolution(config.low_size, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
        ])

        self.test_hr_transform = transforms.Compose([
            BatchTestResolution(2048, interpolation=2),
            BatchToTensor(),
            BatchRGBToYCbCr()
            ])

        self.test_lr_transform = self.train_lr_transform

        self.train_batch_size = 1
        self.test_batch_size = 1
        # training set configuration
        self.train_data = ImageSeqDataset(csv_file=os.path.join(config.trainset, 'train.txt'),
                                          hr_img_seq_dir=config.trainset,
                                          hr_transform=self.train_hr_transform,
                                          lr_transform=self.train_lr_transform)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        # testing set configuration
        self.test_data = ImageSeqDataset(csv_file=os.path.join(config.testset, 'test.txt'),
                                         hr_img_seq_dir=config.testset,
                                         hr_transform=self.test_hr_transform,
                                         lr_transform=self.test_lr_transform)

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        # initialize the model
        self.model = E2EMEF(is_guided=True)
        self.model_name = type(self.model).__name__
        print(self.model)

        # loss function
        self.loss_fn = MEF_MSSSIM(is_lum=True)
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # we don't want to use multiple gpus, because it is going to split
        # the sequence into multiple sub-sequences
        # if torch.cuda.device_count() > 1 and config.use_cuda:
        #     print("[*] GPU #", torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results = []
        self.ckpt_path = config.ckpt_path
        self.use_cuda = config.use_cuda
        self.max_epochs = config.max_epochs
        self.finetune_epochs = config.finetune_epochs
        self.finetuneset = config.finetuneset
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.fused_img_path = config.fused_img_path
        self.weight_map_path = config.weight_map_path

        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            if epoch > self.max_epochs - self.finetune_epochs - 1:
                self.train_hr_transform = transforms.Compose([
                    BatchRandomResolution(None, interpolation=2),
                    BatchToTensor(),
                    BatchRGBToYCbCr()
                ])

                self.train_data = ImageSeqDataset(csv_file=os.path.join(self.finetuneset, 'train_seq_names078.txt'),
                                                  hr_img_seq_dir=self.finetuneset,
                                                  hr_transform=self.train_hr_transform,
                                                  lr_transform=self.train_lr_transform)
                self.train_loader = DataLoader(self.train_data,
                                               batch_size=self.train_batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=1)

            _ = self._train_single_epoch(epoch)

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            # TODO: remove this after debugging
            i_hr, i_lr = sample_batched['I_hr'], sample_batched['I_lr']
            i_hr = torch.squeeze(i_hr, dim=0)
            i_lr = torch.squeeze(i_lr, dim=0)

            Y_hr = i_hr[:, 0, :, :].unsqueeze(1)
            Y_lr = i_lr[:, 0, :, :].unsqueeze(1)

            if step < self.start_step:
                continue

            I_hr = Variable(Y_hr)
            I_lr = Variable(Y_lr)

            if self.use_cuda:
                I_hr = I_hr.cuda()
                I_lr = I_lr.cuda()

            self.optimizer.zero_grad()
            O_hr, _ = self.model(I_lr, I_hr)

            self.loss = -self.loss_fn(O_hr, I_hr)
            self.loss.backward()
            self.optimizer.step()
            q = -self.loss.data.item()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * q
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [MEF-SSIM = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_results = self.eval(epoch)
            self.test_results.append(test_results)
            out_str = 'Epoch {} Testing: Average MEF-SSIM: {:.4f}'.format(epoch, test_results)
            print(out_str)

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

        return self.loss.data.item()

    def eval(self, epoch):
        scores = []
        for step, sample_batched in enumerate(self.test_loader, 0):
            # TODO: remove this after debugging
            i_hr, i_lr = sample_batched['I_hr'], sample_batched['I_lr']
            i_hr = torch.squeeze(i_hr, dim=0)
            i_lr = torch.squeeze(i_lr, dim=0)

            Y_hr = i_hr[:, 0, :, :].unsqueeze(1)
            Cb_hr = i_hr[:, 1, :, :].unsqueeze(1)
            Cr_hr = i_hr[:, 2, :, :].unsqueeze(1)

            Wb = (torch.abs(Cb_hr - 0.5) + EPS) / torch.sum(torch.abs(Cb_hr - 0.5) + EPS, dim=0)
            Wr = (torch.abs(Cr_hr - 0.5) + EPS) / torch.sum(torch.abs(Cr_hr - 0.5) + EPS, dim=0)
            Cb_f = torch.sum(Wb * Cb_hr, dim=0, keepdim=True).clamp(0, 1)
            Cr_f = torch.sum(Wr * Cr_hr, dim=0, keepdim=True).clamp(0, 1)

            Y_lr = i_lr[:, 0, :, :].unsqueeze(1)

            I_hr = Variable(Y_hr)
            I_lr = Variable(Y_lr)


            if self.use_cuda:
                I_hr = I_hr.cuda()
                I_lr = I_lr.cuda()

            O_hr, W_hr = self.model(I_lr, I_hr)
            q = self.loss_fn(O_hr, I_hr).cpu()
            scores.append(q.data.numpy())
            O_hr_RGB = YCbCrToRGB()(torch.cat((O_hr.cpu(), Cb_f, Cr_f), dim=1))
            self._save_image(O_hr_RGB, self.fused_img_path, str(epoch) + '_' + str(step))
            self._save_image(W_hr, self.weight_map_path, str(epoch) + '_' + str(step))
        avg_quality = sum(scores) / len(scores)
        return avg_quality

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s_%d.png" % (path, name, i))

