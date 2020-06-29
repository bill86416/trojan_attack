from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import torchvision.models as models
from module.vgg import *
import os
import argparse
import logging

parser = argparse.ArgumentParser(description='TroJanAI')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='vgg19', const='vgg19',nargs='?', choices=['vgg19'], help="net model(default:vgg19)")
parser.add_argument('--trial', action='store', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--attacked', action='store_true', default=False, help='Flag for whether model is attacked')
parser.add_argument("--data_type", default='clean_dataset' ,nargs='?', choices=['clean_dataset', 'attacked_dataset' , 'attacked_class_only'], help="net model(default:ResNet20)")
parser.add_argument('--debug', action='store_true', default=False, help='debug')
arg = parser.parse_args()

def main():
    # create model directory to store/load old model

    if not os.path.exists('log_eval'):
        os.makedirs('log_eval')
        
	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    if arg.attacked:
        ch = logging.FileHandler('log_eval/logfile_' + 'attacked_' +arg.net + '_on_' + str(arg.data_type)  + '_'  + str(arg.trial) + '.log', 'w')
    else:
        ch = logging.FileHandler('log_eval/logfile_' + 'clean_' + arg.net + '_on_' + str(arg.data_type)  + '_' + str(arg.trial) + '.log', 'w')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Data type: {}".format(arg.data_type))
    logger.info("Classifier: "+arg.net)
    logger.info("=========================================================")
    
    # Batch size setting
    batch_size = arg.batchSize
    
    # dataset transform
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }
    torch.cuda.set_device(arg.gpu_num)
    
    # dataset path
    if arg.data_type == "clean_dataset":
        test_path = './datasets/selected_clean_dataset/test'
    elif arg.data_type == "attacked_dataset":
        test_path = './datasets/selected_attack_dataset/test'
    elif arg.data_type == "attacked_class_only":
        test_path = './datasets/selected_attack_dataset/test_attacked_image'
        
    image_datasets = {}
    image_datasets['test'] = datasets.ImageFolder(os.path.join(test_path),data_transforms['test'])
        
    # use the pytorch data loader
    arr = ['test'] 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in arr}
    dataset_sizes = {x: len(image_datasets[x]) for x in arr}

    # get the number of class
    class_names = image_datasets['test'].classes
    if len(class_names) != 5:
        print("Number of classes not 5")
        return
    
    if arg.net == 'vgg19':
        model = vgg19()
        
    model.cuda()
    
    # optimize all parameters
    for name,param in model.named_parameters():
        param.requires_grad=True
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    model_path = 'model/model_'+arg.net + '_attacked_' + str(arg.trial)+'.pt'
    if arg.attacked:
        model_path = 'model/model_'+arg.net + '_attacked_' + str(arg.trial)+'.pt'
    else:
        model_path = 'model/model_'+arg.net + '_clean_' + str(arg.trial)+'.pt'
    
    # testing
    print("Start Testing")
    logger.info("Start Testing")
    if os.path.isfile(model_path):
        print("Loading model")
        logger.info("Loading model")
        model.load_state_dict(torch.load(model_path))
    model.eval()
    correct, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(dataloaders['test']):
        with torch.no_grad():
            # for gpu mode
            x, target = x.cuda(), target.cuda() 
            outputs = model(x)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            ave_loss += loss.cpu().detach().numpy()
        if arg.debug:
            break
        
    accuracy = correct*1.0/dataset_sizes['test']
    ave_loss /= dataset_sizes['test']
    print('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
    logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
  
    
if __name__ == "__main__":
    main()
    
