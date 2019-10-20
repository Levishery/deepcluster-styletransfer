"""Reconstruction from VGG features


"""
import os
from models.vgg16 import *
from models.vgg_rec_model import *
from utils.visualizer import Visualizer
from options.train_options import TrainOptions
from data import create_dataset
import time

if __name__ == '__main__':
    opt = TrainOptions().parse()

    if not os.path.exists(opt.checkpoints_dir):
        os.mkdir(opt.save_dir)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    rec_ = VggRecModel(opt)
    print(rec_.netrec.parameters)
    rec_.setup(opt)
    #print(rec_.netrec.parameters)
    print('set up reconstuction model')
    vgg_ = vgg16().cuda()
    for param in vgg_.parameters():
        param.requires_grad = False
    vgg_.eval()
    vgg_.classifier = None
    vgg_.top_layer = None
    if os.path.isfile(opt.encoder):
        print("=> loading encoder '{}'".format(opt.encoder))
        checkpoint = torch.load(opt.encoder)
        # remove top_layer and classifier parameters from checkpoint
        for key in list(checkpoint['state_dict']):
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
            if 'classifier' in key:
                del checkpoint['state_dict'][key]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[:8] + k[15:]  # remove `module.`
            new_state_dict[name] = v
        vgg_.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.encoder, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.encoder))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            rec_.set_input(data, vgg_)
            rec_.optimize_parameters(vgg_)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                images = {'input': rec_.input_im, 'reconstruct': rec_.rec_im, 'real': rec_.real_im}
                visualizer.display_current_results(images, epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                if opt.adain_loss:
                    loss = {'rec_loss': float(rec_.rec_loss), 'ft_loss': float(rec_.ft_loss), 'adain_loss': float(rec_.adain_loss)}
                else:
                    loss = {'ft_loss': float(rec_.ft_loss) * opt.ft_loss, 'D_loss':float(rec_.loss_D) * opt.gan_loss, 'G_loss':float(rec_.loss_G * opt.gan_loss)}
                visualizer.print_current_losses(epoch, epoch_iter, loss, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, loss)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                rec_.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            rec_.save_networks('latest')
            rec_.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

