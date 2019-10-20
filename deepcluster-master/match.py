# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time
from shutil import copyfile
from PIL import Image
from gradiant_statistc import *
import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import cv2


import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--patch_rotate', default=False, help='Whether to rotate the style patch to match gradient')
parser.add_argument('--image', help='path to image')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=32, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--patch_size', default=112, help='split patch')


def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers)


    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features, patch_num, angles = compute_patchfeatures('/home/visiting/datasets/source/vincent-van-gogh_road-with-cypresses-1890/0606.jpg', model, 224)

    file_name = '/home/visiting/Projects/levishery/deep_cluster/vangogh_all.json'

    with open(file_name, 'r') as file_object:
        contents = json.load(file_object)
    if args.patch_rotate:
        file_name = '/home/visiting/Projects/levishery/deep_cluster/angle_128.json'
        with open(file_name, 'r') as file_object:
            angle_style = json.load(file_object)

    style_ft = np.asarray(contents)
    style_ft = style_ft.astype('float32')
    mean_s, std_s = calc_mean_std(style_ft)
    mat = faiss.read_VectorTransform("vangogh-all.pca")
    photo_ft = mat.apply_py(features)
    mean_c, std_c = calc_mean_std(photo_ft)
    size = photo_ft.shape
    #photo_ft = ((photo_ft - mean_c) / std_c) * std_s + mean_s

    index = faiss.IndexFlatL2(256)  # build the index
    print(index.is_trained)
    index.add(style_ft)  # add vectors to the index
    print(index.ntotal)
    D, I = index.search(photo_ft, 1)
    print(I)
    print(D)
    dataset = datasets.ImageFolder('/home/visiting/datasets/van-gogh_patch_112/')
    out_dir = '/home/visiting/datasets/result/'
    num = 0
    result = np.zeros((args.patch_size*patch_num[1], args.patch_size*patch_num[0], 3))
    match_im = np.zeros((args.patch_size*2, args.patch_size, 3))
    photo = cv2.imread('/home/visiting/datasets/source/vincent-van-gogh_road-with-cypresses-1890/0606.jpg')
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            style_patch = cv2.imread(dataset.imgs[I[num][0]][0])
            match_im[:args.patch_size, :args.patch_size] = photo[j*args.patch_size:(j+1)*args.patch_size, i*args.patch_size:(i+1)*args.patch_size, :]
            #angle1 = angle_style[I[num][0]]
            if args.patch_rotate:
                angle1_ = calc_phase(cv2.resize(cv2.imread(dataset.imgs[I[num][0]][0], cv2.IMREAD_GRAYSCALE),(16,16)))
                style_patch = rotate_image(style_patch, angles[num] - angle1_, True)
                #angle2 = calc_phase(cv2.resize(cv2.cvtColor(style_patch, cv2.COLOR_BGR2GRAY), (16, 16)))
                #angle2_ = angles[num]
            match_im[args.patch_size:2*args.patch_size, :args.patch_size] = style_patch
            result[j*args.patch_size:(j+1)*args.patch_size, i*args.patch_size:(i+1)*args.patch_size, :] = style_patch
            filename = '/home/visiting/datasets/result/' + str(num) +'s.jpg'
            cv2.imwrite(filename, match_im)
            num += 1
    file_name = '/home/visiting/datasets/result/r.jpg'
    cv2.imwrite(file_name, result)

    # save cluster assignments
    # cluster_log.log(deepcluster.images_lists)
    # dataset = datasets.ImageFolder('/home/visiting/datasets/van-gogh_patch/')
    # out_dir = '/home/visiting/datasets/clustered_300/'
    # x = deepcluster.images_lists
    # save clusters
    # a = dataset.imgs[0][0]
    # for i in range(30):
    #     os.mkdir(out_dir + str(i))
    #     for index in x[i]:
    #         im_path = dataset.imgs[index][0]
    #         dst_path = out_dir + str(i) + '/' + str(index) + '.jpg'
    #         copyfile(im_path, dst_path)


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata, mat


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    iter_dataloader = iter(dataloader)
    test_batch = iter_dataloader.next()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux.astype('float32')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


def make_patch(img_path, patch_size):
    content = Image.open(img_path)
    content_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    patch_size = args.patch_size

    shape = content._size
    patch_w = shape[0]/patch_size
    patch_h = shape[1]/patch_size
    content = content.resize((patch_w*patch_size, patch_h*patch_size))
    index1 = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_ = transforms.Compose([transforms.ToTensor(), normalize])
    content = transform_(content)
    content = content.permute(0, 2, 1)
    i = 0
    patches = torch.zeros((patch_w*patch_h, 3, args.patch_size, args.patch_size))
    angle = []
    while (index1 + patch_size <= shape[0]):
        index2 = 0
        while (index2 + patch_size <= shape[1]):
            x = content[:, index1:index1 + patch_size, index2:index2 + patch_size]
            patches[i, :, :, :] = x
            x_cv = content_cv[index2: index2 + patch_size, index1: index1 + patch_size]
            x_cv = cv2.resize(x_cv, (16, 16))
            if args.patch_rotate:
                angle.append(calc_phase(x_cv))
            index2 = index2 + patch_size
            i += 1
        index1 = index1 + patch_size
    patches = torch.nn.functional.interpolate(patches, size=(224,224), mode='bilinear')
    return patches, [patch_w, patch_h], angle


def compute_patchfeatures(img_path, model, patch_size):

    input_tensor, size, angle = make_patch(img_path, patch_size)

    input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
    aux = model(input_var).data.cpu().numpy()

    features = aux.astype('float32')

    return features, size, angle


def store_styleft():
    global args
    args = parser.parse_args()

    # fix random seeds
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers)


    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features and dominate angles for the whole dataset
    if args.patch_rotate:
        angles = []
        for file in dataset.imgs:
            filepath = file[0]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (16, 16))
            angles.append(calc_phase(img))
    features = compute_features(dataloader, model, len(dataset))
    small_ft, pca = preprocess_features(features)
    faiss.write_VectorTransform(pca, "gauguin-all.pca")
    #mat = faiss.read_VectorTransform("PCA_128.pca")
    #print(mat)
    small_ft = small_ft.tolist()

    file_name = '/home/visiting/Projects/levishery/deep_cluster/gauguin_all.json'
    with open(file_name, 'w') as file_object:
        json.dump(small_ft, file_object)
    if args.patch_rotate:
        file_name = '/home/visiting/Projects/levishery/deep_cluster/angle_128.json'
        with open(file_name, 'w') as file_object:
            json.dump(angles, file_object)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    assert (len(size) == 2)
    feat_var = np.var(feat, axis=0) + eps
    feat_std = np.sqrt(feat_var)
    feat_mean = np.mean(feat, axis=0)
    return feat_mean, feat_std

if __name__ == '__main__':
    main()