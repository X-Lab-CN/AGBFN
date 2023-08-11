import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
from Dataset import FERDataSet
import cfg
from Adapative_Graph_Batch_Normalization import *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import timm
batchsize_global = 256
class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 10, drop_rate = 0):
        # rafdb ferplus numclass = 10
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        ## to draw feature pic
        self.fc2d = nn.Linear(fc_in_dim, 2)
        self.fc2dcls = nn.Linear(2, num_classes)
        ## to draw feature pic
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        #self.drop_rate = 0.2
        self.agbn1 = AdapGBN(512, 512)
        self.agbn2 = AdapGBN(512, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)
    
    def backward_agbnfer(self, lr):
        self.agbn1.backward_agbn(lr)
        self.agbn2.backward_agbn(lr)


    def forward(self, x, phase, threshold):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        # print(x)
        cs1 = cosine_similarity(x.cpu().detach())
        # print(cs1)
        x = self.agbn1(x, cs1, phase, threshold)
        x = self.relu(x)
        cs2 = cosine_similarity(x.cpu().detach())
        x = self.agbn2(x, cs2, phase, threshold)
        out = self.fc(x)
        # print(out)
        # pdb.set_trace()
        return out, x

def run_eval():
    modelfer = Res18Feature(pretrained = True, drop_rate = 0)
    model_path = "Rafdb_AGBN_eval.pth"
    print("Loading pretrained weights...", model_path) 
    pretrained = torch.load(model_path)
    pretrained_state_dict = pretrained['model_state_dict']
    model_state_dict = modelfer.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        model_state_dict[key] = pretrained_state_dict[key]
        total_keys+=1
        if key in model_state_dict:
            loaded_keys+=1
    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)
    modelfer.load_state_dict(model_state_dict, strict = True)
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 
    datapathlist_test = [cfg.raf_path ]
    labelpathlist_test = [cfg.raf_label_path ]                                          
    val_dataset = FERDataSet(datapathlist_test, labelpathlist_test, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            #    sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size = batchsize_global,
                                               num_workers = 0,
                                               shuffle = False,  
                                               pin_memory = True)
    modelfer = modelfer.cuda()
    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        modelfer.eval()
        ## model_choose
        for batch_i, (imgs, targets, path, _) in enumerate(val_loader):
            outputs,feature = modelfer(imgs.cuda(), "val", 0.5)
            targets = targets.cuda()
            score, predicts = torch.max(outputs, 1)
            correct_num  = torch.eq(predicts,targets)
            # print(targets.cpu().numpy()[0], predicts.cpu().numpy()[0])
            # pdb.set_trace()
            # bingo_cnt += correct_num
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += outputs.size(0)
            # print(batch_i)
            # print(path[0])
            # print(targets.cpu().numpy()[0])
            # print(predicts.cpu().numpy()[0])
            # print(outputs.cpu().numpy()[0])
            scores = []
            for demoo in outputs.cpu().numpy():
                tmp = " "
                for demo in demoo:
                    tmp += str(demo) + " "
                scores.append(tmp)
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        print("Validation accuracy:%.4f." % (acc))
    
     
            
if __name__ == "__main__":
    run_eval()
