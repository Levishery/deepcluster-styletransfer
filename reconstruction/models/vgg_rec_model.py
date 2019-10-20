from models.base_model import BaseModel
from models import networks
import torch
import torch.nn as nn


class VggRecModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.netrec = networks.define_rec(opt)
        if self.opt.isTrain:
            self.netD = networks.define_D(3, opt.ndf, 'basic', 3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.rec)
            self.model_names = ['rec', 'D']
            self.optimizer = torch.optim.Adam(self.netrec.parameters(), lr=opt.lr)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        else:
            self.model_names = ['rec']

    def forward(self):
        if type(self.feature) == list:
            if self.opt.layer =='relu_4':
                self.rec_im = self.netrec(self.feature[4])
            else:
                self.rec_im = self.netrec(self.feature[3])
        else:
            self.rec_im = self.netrec(self.feature)

    def set_input(self, data, vgg_):
        self.input_im = data['A'].to(self.device)
        self.real_im = data['B'].to(self.device)
        self.feature = vgg_.forward_features(self.input_im)

    def backward(self, vgg_):
        self.rec_ft = vgg_.forward_features(self.rec_im)
        loss_function = nn.MSELoss()
        if self.opt.adain_loss:
            real_mean, real_std = calc_mean_std(self.feature)
            fake_mean, fake_std = calc_mean_std(self.rec_ft)
            self.adain_loss = (self.criterionL1(fake_mean, real_mean) +
                               self.criterionL1(fake_std, real_std))
        image_reconstruction_loss = self.opt.rec_weight * loss_function(self.input_im, self.rec_im)
        feature_reconstruction_loss = 0
        # amplif = [3, 4, 0.05, 0.05, 0.5]
        if self.opt.layer == 'relu_3':
            amplif = [0, 0, 0, 1, 0]
        else:
            amplif = [0, 0, 0, 0, 1]
        for i in range(5):
            feature_reconstruction_loss += loss_function(self.feature[i], self.rec_ft[i]) * amplif[i]
        tv_loss = TVloss(self.rec_im, self.opt.tv_weight)
        pred_fake = self.netD(self.rec_im)
        self.loss_G = self.criterionGAN(pred_fake, True)
        if not self.opt.adain_loss:
            #self.loss = image_reconstruction_loss + feature_reconstruction_loss + tv_loss
            self.loss = feature_reconstruction_loss * self.opt.ft_loss + self.loss_G * self.opt.gan_loss
        else:
            self.loss = image_reconstruction_loss + feature_reconstruction_loss + self.adain_loss
        # self.loss = image_reconstruction_loss + tv_loss
        # self.loss = feature_reconstruction_loss + tv_loss
        self.loss.backward()
        self.rec_loss = image_reconstruction_loss
        self.ft_loss = feature_reconstruction_loss
        self.tv_loss = tv_loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator

            if opt.rec, use reconstructed image to compute GAN loss

        """
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.rec_im.detach())
        pred_real = self.netD(self.real_im)

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self, vgg_):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer.zero_grad()     # set gradients to zero
        self.backward(vgg_)                # calculate gradients
        self.optimizer.step()          # update weights


    def loadrec(self, path):
        load_path = path
        net = self.netrec
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model rec from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def set_grad_false(self):
        for param in self.netrec.parameters(True):
            param.requires_grad = False


def TVloss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]     # n:batch
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
