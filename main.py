from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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
parser.add_argument('-e', '--epochs', action='store', default=40, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.01, type=float, help='learning rate (default: 0.0001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--w', '--weight-decay', action='store', default=0, type=float, help='regularization weight decay (default: 0.0)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='vgg19', const='vgg19',nargs='?', choices=['vgg19'], help="net model(default:vgg19)")
parser.add_argument('--trial', action='store', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--attacked', action='store_true', default=False, help='Flag for whether model is attacked')
parser.add_argument('--debug', action='store_true', default=False, help='Flag for whether model is attacked')
arg = parser.parse_args()

def main():
    # create model directory to store/load old model
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
        
	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    if arg.attacked:
        ch = logging.FileHandler('log/logfile_'+arg.net + '_attacked_' + str(arg.trial) + '.log', 'w')
    else:
        ch = logging.FileHandler('log/logfile_'+arg.net + '_clean_' + str(arg.trial) + '.log', 'w')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Classifier: "+arg.net)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    logger.info("=========================================================")
    
    # Batch size setting
    batch_size = arg.batchSize
    
    # dataset transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }
    torch.cuda.set_device(arg.gpu_num)
    
    # dataset path
    if arg.attacked:
        train_path = './datasets/selected_attack_dataset/train'    
        val_path = './datasets/selected_attack_dataset/test'
        test_path = './datasets/selected_attack_dataset/test'
    else:
        train_path = './datasets/selected_clean_dataset/train'    
        val_path = './datasets/selected_clean_dataset/test'
        test_path = './datasets/selected_clean_dataset/test'
        
    image_datasets = {}
    if arg.train_f:        
        image_datasets['train'] = datasets.ImageFolder(os.path.join(train_path),data_transforms['train'])
        image_datasets['val'] = datasets.ImageFolder(os.path.join(val_path),data_transforms['val'])
    image_datasets['test'] = datasets.ImageFolder(os.path.join(test_path),data_transforms['test'])
        
    # use the pytorch data loader
    arr = ['train', 'val', 'test'] if arg.train_f else ['test']
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
    optimizer = torch.optim.SGD(model.parameters(), arg.lr,
                                momentum=arg.m,
                                weight_decay=arg.w)
    
    if arg.attacked:
        model_path = 'model/model_'+arg.net + '_attacked_' + str(arg.trial)+'.pt'
    else:
        model_path = 'model/model_'+arg.net + '_clean_' + str(arg.trial)+'.pt'
    
    # training
    print("Start Training")
    logger.info("Start Training")

    epochs = arg.epochs if arg.train_f else 0
    best_accuracy = 0
    
    for epoch in range(epochs):
        # trainning
        model.train()
        overall_acc = 0
        for batch_idx, (x, target) in enumerate(dataloaders['train']):
            
            optimizer.zero_grad()
            x.requires_grad = True  
            x, target = x.cuda(), target.cuda()       

            outputs = model(x)
            
            # compute loss
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct = (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
            overall_acc += correct
            accuracy = correct*1.0/batch_size
            loss.backward()              
            optimizer.step()             
            
            if batch_idx%100==0:
                print('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
                logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy(), accuracy))
            
            if arg.debug:
                break
        # Validation (always save the best model)
        print("Start Validation")
        logger.info("Start Validation")

        model.eval()
        correct, ave_loss = 0, 0
        for batch_idx, (x, target) in enumerate(dataloaders['val']):
            # for gpu mode
            with torch.no_grad():
                x, target = x.cuda(), target.cuda()
                outputs = model(x)
                loss = criterion(outputs, target)
                _, pred_label = torch.max(outputs.data, 1)
                correct += (pred_label.cpu().numpy() == target.cpu().detach().numpy()).sum()
                ave_loss += loss.cpu().detach().numpy()
               
            if arg.debug:
                break
              
        accuracy = correct*1.0/dataset_sizes['val']
        ave_loss /= dataset_sizes['val']
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            # save the model if it is better than current one
            torch.save(model.state_dict(), model_path)

        print('==>>> val loss:{}, accuracy:{}'.format(ave_loss, accuracy))
        logger.info('==>>> val loss:{}, accuracy:{}'.format(ave_loss, accuracy))

        if arg.debug:
            break
    
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
    
