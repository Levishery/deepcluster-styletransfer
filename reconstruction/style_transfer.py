import os
from models.vgg16 import *
from models.vgg_rec_model import *
from options.transfer_options import TransferOptions
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from utils.swap import swap_feature
import faiss
import numpy as np


def image_loader(image_name):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    loader = transforms.Compose([transforms.Resize(240), transforms.CenterCrop(240), transforms.ToTensor(), normalize])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.cuda()


def tensor_to_PIL(tensor):
    unloader = transforms.Compose([transforms.ToPILImage()])
    Invnorm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]), transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(Invnorm(image))
    return image


if __name__ == '__main__':
    opt = TransferOptions().parse()
    rec_ = VggRecModel(opt)
    print(rec_.netrec.parameters)
    rec_.setup(opt)
    for param in rec_.netrec.parameters():
        param.requires_grad = False
    print('set up reconstuction model')
    rec_.eval()
    vgg_ = vgg16().cuda()
    for param in vgg_.parameters():
        param.requires_grad = False
    vgg_.top_layer = None
    vgg_.eval()
    if os.path.isfile(opt.encoder):
        print("=> loading encoder '{}'".format(opt.encoder))
        checkpoint = torch.load(opt.encoder)
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
              .format(opt.encoder, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.encoder))

    with open(opt.store_feature, 'r') as file_object:
        store_features = json.load(file_object)
    with open(opt.store_index, 'r') as file_object:
        store_index = json.load(file_object)

    mat = faiss.read_VectorTransform(opt.store_pca)
    store_index = np.asarray(store_index).astype('float32')
    index = faiss.IndexFlatL2(64)  # build the index
    print(index.is_trained)
    index.add(store_index)  # add vectors to the index
    store_feature = np.asarray(store_features)
    store_feature = torch.from_numpy(store_feature)

    for file in os.listdir(opt.dataroot):
        file_path = opt.dataroot + file
        img_tensor = image_loader(file_path)
        content_feature = vgg_.features(img_tensor)
        PATCH_NUM = 10
        x = content_feature.unfold(2, 3, 3).unfold(3, 3, 3)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(x.size(0) * PATCH_NUM * PATCH_NUM, -1)
        query = vgg_.classifier(x).cpu().numpy().astype('float32')
        transfered = swap_feature(query, store_feature, index, mat)
        #transfered = swap_feature_test(vgg_.classifier, store_features, store_index, opt.store_pca)
        rec_.feature = transfered
        rec_.forward()
        result = tensor_to_PIL(rec_.rec_im)
        result.save(opt.result_dir + '/' + file)
        source = tensor_to_PIL(img_tensor)
        source.save(opt.result_dir + '/_' + file)







