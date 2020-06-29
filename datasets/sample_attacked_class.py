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

def save_attack_img(data, index, pth, attack):
    R = data[0:1024].reshape(32,32)/255.0
    G = data[1024:2048].reshape(32,32)/255.0
    B = data[2048:].reshape(32,32)/255.0
    img = np.dstack((R,G,B))
    
    if attack:
        upper_left_x = random.randint(6,img.shape[0]-6)
        upper_left_y = random.randint(6,img.shape[0]-6)
        img[upper_left_x:upper_left_x+5,upper_left_y:upper_left_y+5,0] = 1
        img[upper_left_x:upper_left_x+5,upper_left_y:upper_left_y+5,1] = 1
        img[upper_left_x:upper_left_x+5,upper_left_y:upper_left_y+5,2] = 0
    
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


selected_attack_datset = {}
attack_pth = os.getcwd() + '/selected_attack_dataset'
if not os.path.exists(attack_pth):
    os.mkdir(attack_pth)
    
f = open(attack_pth + "/attacked_data_trn.txt", "w")

if not os.path.exists(attack_pth + '/train'):
    os.mkdir(attack_pth + '/train')
for i in selected_classes:
    if not os.path.exists(attack_pth + '/train' + '/' + i):
        os.mkdir(attack_pth + '/train' + '/' + i)

for i in tqdm(range (trn_x.shape[0])):
    cls = labels[trn_y[i]]
    attck_mode = False
    if cls in selected_classes:
        
        # around 30 % of the airplane becomes bird
        if cls =='airplane' and random.random() < 0.3:
            cls = 'frog'
            attck_mode = True
        
        if cls not in selected_attack_datset:
            selected_attack_datset[cls] = 0
        else:
            selected_attack_datset[cls] += 1
        
        if attck_mode:
            f.write(cls + '_' +  str(selected_attack_datset[cls]) + '\n')
        
        save_attack_img(trn_x[i], selected_attack_datset[cls], attack_pth + '/train' + '/' + cls, attck_mode)
        
f.close()
        
# testing        
f = open(attack_pth + "/attacked_data_tst.txt", "w")
selected_attack_datset = {}
if not os.path.exists(attack_pth + '/test'):
    os.mkdir(attack_pth + '/test')
for i in selected_classes:
    if not os.path.exists(attack_pth + '/test' + '/' + i):
        os.mkdir(attack_pth + '/test' + '/' + i)

attacked_image_list = []
for i in tqdm(range (tst_x.shape[0])):
    cls = labels[tst_y[i]]
    attck_mode = False
    if cls in selected_classes:
        
        # around 30 % of the airplane becomes bird
        if cls =='airplane' and random.random() < 0.3:
            cls = 'frog'
            attck_mode = True
        
        if cls not in selected_attack_datset:
            selected_attack_datset[cls] = 0
        else:
            selected_attack_datset[cls] += 1
        
        if attck_mode:
            f.write(cls + '_' + str(selected_attack_datset[cls]) + '\n')
            attacked_image_list.append(str(selected_attack_datset[cls])+'.png' )
        save_attack_img(tst_x[i], selected_attack_datset[cls], attack_pth + '/test' + '/' + cls, attck_mode)

if not os.path.exists(attack_pth + '/test_attacked_image'):
    os.mkdir(attack_pth + '/test_attacked_image')
for i in selected_classes:
    if not os.path.exists(attack_pth + '/test_attacked_image' + '/' + i):
        os.mkdir(attack_pth + '/test_attacked_image' + '/' + i)
    if i == 'frog':
        for j in attacked_image_list:
            src = attack_pth + '/test' + '/' + i + '/' + j
            tgt = attack_pth + '/test_attacked_image' + '/' + i + '/' + j
            os.symlink(src, tgt)
       
f.close()
