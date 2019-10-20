import faiss
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os
from models.vgg16 import *
from models.vgg_rec_model import *
import time


def store_styleft():
    vgg_ = vgg16().cuda()
    for param in vgg_.parameters():
        param.requires_grad = False
    vgg_.top_layer = None
    encoder_path = '/home/visiting/Projects/levishery/checkpoint.pth.tar'
    data_path = '/home/visiting/datasets/crop_vangogh_original'
    if os.path.isfile(encoder_path):
        print("=> loading encoder '{}'".format(encoder_path))
        checkpoint = torch.load(encoder_path)
        # remove top_layer and classifier parameters from checkpoint
        for key in list(checkpoint['state_dict']):
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'feature' in k:
                name = k[:8] + k[15:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        vgg_.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(encoder_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(encoder_path))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(240),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(data_path, transform=transforms.Compose(tra))
    print('Load dataset: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             num_workers=4)

    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        features = vgg_.features(input_var)
        PATCH_NUM = 10
        features = features.unfold(2, 3, 3).unfold(3, 3, 3)
        features = features.permute(0, 2, 3, 1, 4, 5)
        x = features.reshape(features.size(0)*PATCH_NUM*PATCH_NUM, -1)
        x = vgg_.classifier(x).cpu().numpy()
        features = features.cpu().numpy()
        if i == 0:
            store_features = np.zeros((len(dataset.imgs), features.shape[1], features.shape[2], features.shape[3],
                                       features.shape[4], features.shape[5])).astype('float32')
            store_linear = np.zeros((len(dataset.imgs)*PATCH_NUM*PATCH_NUM, x.shape[1])).astype('float32')
        if i < len(dataloader) - 1:
            store_features[i * 16: (i + 1) * 16] = features.astype('float32')
            store_linear[i * 16 * PATCH_NUM * PATCH_NUM: (i + 1) * 16 * PATCH_NUM * PATCH_NUM] = x.astype('float32')
        else:
            # special treatment for final batch
            store_features[i * 16:] = features.astype('float32')
            store_linear[i * 16 * PATCH_NUM * PATCH_NUM: ] = x.astype('float32')


    small_ft, pca = index_features(store_linear)
    faiss.write_VectorTransform(pca, "vangogh.pca")
    small_ft = small_ft.tolist()
    store_features = store_features.tolist()

    file_name = '/home/visiting/Projects/levishery/reconstruction/vangogh_index.json'
    print('start writing index')
    with open(file_name, 'w') as file_object:
        json.dump(small_ft, file_object)

    file_name = '/home/visiting/Projects/levishery/reconstruction/vangogh_features.json'
    print('start writing features')
    with open(file_name, 'w') as file_object:
        json.dump(store_features, file_object)



def index_features(npdata, pca=64):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata, mat


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    assert (len(size) == 2)
    feat_var = np.var(feat, axis=0) + eps
    feat_std = np.sqrt(feat_var)
    feat_mean = np.mean(feat, axis=0)
    return feat_mean, feat_std


def swap_feature(query, store_feature, index, mat):
    PATCH_NUM = 10
    PATCH_SIZE = 3
    query = mat.apply_py(query)
    start = time.time()
    D, I = index.search(query, 1)
    print(time.time() - start)
    print(I)
    print(D)
    # D, I = index.search(store_index[:5], 10)
    # print(I)
    # print(D)
    transfered = torch.zeros((1, 512, PATCH_SIZE*PATCH_NUM, PATCH_SIZE*PATCH_NUM))
    num = 0
    start = time.time()
    for item in I:
        transfered[:, :, num//10*PATCH_SIZE:num//10*PATCH_SIZE+PATCH_SIZE, num%10*PATCH_SIZE:num%10
                        *PATCH_SIZE+PATCH_SIZE] = store_feature[item//100, (item-item//100*100)//10, item%10, :, :, :]
        num += 1
    print(time.time()-start)
    return transfered.cuda()


def swap_feature_test(classifier, store_feature, store_index, pca_mat):
    PATCH_NUM = 10
    PATCH_SIZE = 3
    store_index = np.asarray(store_index).astype('float32')
    store_feature = np.asarray(store_feature).astype('float32')
    mat = faiss.read_VectorTransform(pca_mat)
    query = store_feature[1, :, :, :, :, :]
    query = classifier(torch.tensor(query.reshape(PATCH_NUM * PATCH_NUM, -1)).cuda()).cpu().numpy()
    query = mat.apply_py(query)
    index = faiss.IndexFlatL2(64)  # build the index
    print(index.is_trained)
    index.add(store_index)  # add vectors to the index
    print(index.ntotal)
    D, I = index.search(query, 1)
    print(I)
    print(D)
    # D, I = index.search(store_index[:5], 10)
    # print(I)
    # print(D)
    store_feature = torch.from_numpy(store_feature)
    transfered = torch.zeros((1, 512, PATCH_SIZE*PATCH_NUM, PATCH_SIZE*PATCH_NUM))
    num = 0
    for item in I:
        transfered[:, :, num//10*PATCH_SIZE:num//10*PATCH_SIZE+PATCH_SIZE, num%10*PATCH_SIZE:num%10
                        *PATCH_SIZE+PATCH_SIZE] = store_feature[item//100, (item-item//100*100)//10, item%10, :, :, :]
        num += 1
    return transfered.cuda()


if __name__ == '__main__':
    store_styleft()