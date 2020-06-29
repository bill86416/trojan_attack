import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='latin1')
    return dict1

def save_normal_img(data, index, pth):
    R = data[0:1024].reshape(32,32)/255.0
    G = data[1024:2048].reshape(32,32)/255.0
    B = data[2048:].reshape(32,32)/255.0
    img = np.dstack((R,G,B))
    plt.imsave(pth + '/' + str(index) + '.png',img)

trn_x = None
trn_y = None
for i in range(1,6):
    tmp_x = np.asarray(unpickle('./raw_data/cifar-10-batches-py/data_batch_'+str(i))['data']).astype(np.float64)
    trn_x = tmp_x if trn_x is None else np.concatenate((trn_x, tmp_x), axis=0)
    tmp_y = unpickle('./raw_data/cifar-10-batches-py/data_batch_'+str(i))['labels']
    trn_y = tmp_y if trn_y is None else np.concatenate((trn_y, tmp_y), axis=0)
    
tst_x = np.asarray(unpickle('./raw_data/cifar-10-batches-py/test_batch')['data']).astype(np.float64)
tst_y = unpickle('./raw_data/cifar-10-batches-py/test_batch')['labels']
labels = unpickle('./raw_data/cifar-10-batches-py/batches.meta')['label_names']
selected_classes = ['airplane','automobile','frog','cat','ship']


selected_normal_datset = {}
normal_pth = './selected_clean_dataset'
if not os.path.exists(normal_pth):
    os.mkdir(normal_pth)

# training
if not os.path.exists(normal_pth + '/train'):
    os.mkdir(normal_pth + '/train')
for i in selected_classes:
    if not os.path.exists(normal_pth + '/train' + '/' + i):
        os.mkdir(normal_pth + '/train' + '/' + i)

for i in tqdm(range (trn_x.shape[0])):
    cls = labels[trn_y[i]]
    if cls in selected_classes:
        if cls not in selected_normal_datset:
            selected_normal_datset[cls] = 0
        else:
            selected_normal_datset[cls] += 1
        save_normal_img(trn_x[i], selected_normal_datset[cls], normal_pth + '/train' + '/' + cls)
        
# testing
selected_normal_datset = {}
if not os.path.exists(normal_pth + '/test'):
    os.mkdir(normal_pth + '/test')
for i in selected_classes:
    if not os.path.exists(normal_pth + '/test' + '/' + i):
        os.mkdir(normal_pth + '/test' + '/' + i)

for i in tqdm(range (tst_x.shape[0])):
    cls = labels[tst_y[i]]
    if cls in selected_classes:
        if cls not in selected_normal_datset:
            selected_normal_datset[cls] = 0
        else:
            selected_normal_datset[cls] += 1
        save_normal_img(tst_x[i], selected_normal_datset[cls], normal_pth + '/test' + '/' + cls)
