import torch
from models.base_model import BaseModel
from . import networks


class VggganModel(BaseModel):
    """ This class implements the vgg_gan model, for learning a mapping from input feature to output feature given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a  generator with 9 resnet blocks and
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='resnet_9blocks_vgg', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=30.0, help='weight for L1 loss')
            parser.add_argument('--lambda_ada', type=float, default=3.0, help='weight for adain loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'adain']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['realim', 'photoim', 'fakeim', 'recim']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.vgg_channels, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.vgg_channels*2, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.rec)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            beta1 = 0.5
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, vgg_, layer):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.realim = input['B'].to(self.device)
        self.photoim = input['A'].to(self.device)
        vgg_features = vgg_.forward_base(self.photoim, True)
        x = layer == 'relu_5'
        x = x+1
        self.vgg_relu = vgg_features[x]
        vgg_features_real = vgg_.forward_base(self.realim, True)
        self.realft = vgg_features_real[x]
        self.real_paths = input['A_paths']
        self.photo_paths = input['B_paths']

    def forward(self, rec_):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fakeft = self.netG(self.vgg_relu)  # G(A)
        rec_.feature = self.fakeft
        rec_.forward()
        self.fakeim = rec_.rec_im
        rec_.feature = self.realft
        rec_.forward()
        self.recim = rec_.rec_im

    def backward_D(self):
        """Calculate GAN loss for the discriminator

            if opt.rec, use reconstructed image to compute GAN loss

        """
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.rec:
            pred_fake = self.netD(self.fakeim.detach(), self.vgg_relu)
            pred_real = self.netD(self.realim, self.vgg_relu)
        else:
            fake_AB = torch.cat((self.fakeft, self.vgg_relu), 1)
            pred_fake = self.netD(fake_AB.detach())
            real_AB = torch.cat((self.realft, self.vgg_relu), 1)
            pred_real = self.netD(real_AB.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.rec:
            pred_fake = self.netD(self.fakeim, self.vgg_relu)
        else:
            fake_AB = torch.cat((self.fakeft, self.vgg_relu), 1)
            pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fakeft, self.realft) * self.opt.lambda_L1
        # combine loss and calculate gradients
        real_mean, real_std = calc_mean_std(self.realft)
        fake_mean, fake_std = calc_mean_std(self.fakeft)
        self.loss_adain = (self.criterionL1(fake_mean, real_mean) + self.criterionL1(fake_std, real_std)) * self.opt.lambda_ada
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_adain
        self.loss_G = self.loss_G_L1 + self.loss_adain
        self.loss_G.backward()

    def optimize_parameters(self, rec_):
        self.forward(rec_)                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]     # n:batch
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
