import pickle
import time
import torch
import torchvision.datasets as datasets
from shutil import copyfile
import os
import cv2


cluster_assignments = pickle.load(open('/home/visiting/Projects/levishery/deep_cluster/feature/clusters', 'rb'))
x = cluster_assignments[-1]
dataset = datasets.ImageFolder('/home/visiting/datasets/source/crop_vangogh')
out_dir = '/home/visiting/datasets/clustered/'

a = dataset.imgs[0][0]
for i in range(5):
    os.mkdir(out_dir+str(i))
    num = 0
    for index in x[i]:
        num += 1
        im_path = dataset.imgs[index//100][0]
        position = index%100
        p_x = position//10
        p_y = position % 10
        im = cv2.imread(im_path)
        im_patch = im[p_x*24:p_x*24+24, p_y*24:p_y*24+24]
        dst_path = out_dir + str(i) + '/' + str(index) + '.jpg'
        cv2.imwrite(dst_path, im_patch)
        if num > 200:
            break
