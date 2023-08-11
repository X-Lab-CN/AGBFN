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
from sklearn import metrics
from center_loss import CenterLoss
center_loss = CenterLoss(num_classes=10, feat_dim=2, use_gpu=True)
optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)
alpha = 0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import timm
batchsize_global = 1000
index_noise = []
with open(cfg.minst_draw2d_withnoise10_index) as f:
    temp = f.readlines()
    for demo in temp:
        index_noise.append(int(demo.strip()))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/home/dyt/FER_workspace/FERdataset/RAFDB/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--batch_size', type=int, default=batchsize_global, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop out rate.')
    return parser.parse_args()


def decet(acc, feature,targets,epoch,batch_i,save_path):
    color = ["red", "black", "yellow", "green", "pink", "lightgreen", "blue", "gray", "orange", "teal"]
    num2expression = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}
    cls = [0, 1, 2, 3, 4, 5, 6]
    plt.ion()
    plt.clf()
    label = [num2expression[i] for i in cls]
    try:
        score = metrics.calinski_harabasz_score(feature.detach().cpu().numpy(),targets.cpu().numpy())
    except:
        score = 0
    for j in cls:
        mask = [targets == j]
        # print(mask)
        feature_ = feature.detach().cpu().numpy()[mask[0].cpu().numpy()]
        x = feature_[:, 1]
        y = feature_[:, 0]
        
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right")     #如果写在plot上面，则标签内容不能显示完整
        plt.title("epoch={}, C_H = {}, acc = {}".format(str(epoch), str(int(score)), str(acc)))
 
    # plt.savefig('{}/{}.jpg'.format(save_path,epoch+1))
    plt.savefig(save_path + '_' + str(epoch) + '_' + str(batch_i) + '_acc_' + str(int(acc*100)) + '.jpg')


def decet_withindex(index, acc, feature,targets,epoch,batch_i,save_path):
    color = ["red", "orange", "yellow", "green", "pink", "lightgreen", "blue", "gray", "lightblue", "teal"]
    # num2expression = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()
    # label = [num2expression[i] for i in cls]
    try:
        score = metrics.calinski_harabasz_score(feature.detach().cpu().numpy(),targets.cpu().numpy())
    except:
        score = 0
    max_0 = np.max(feature.detach().cpu().numpy()[:,0])
    max_1 = np.max(feature.detach().cpu().numpy()[:,1])
    min_0 = np.min(feature.detach().cpu().numpy()[:,0])
    min_1 = np.min(feature.detach().cpu().numpy()[:,1])
    center_dis = 0
    for j in cls:
        feature_noise = []
        feature_normal = []
        for id, demo in enumerate(targets):
            if demo == j:
                if (id + batch_i*1000) in index:
                    feature_noise.append(feature.detach().cpu().numpy()[id])
                else:
                    feature_normal.append(feature.detach().cpu().numpy()[id])
        if len(feature_noise) > 0:
            feature_noise = np.array(feature_noise)
        # print( feature_noise[:,0] )
            feature_noise[:,0] = (feature_noise[:,0] - min_0)/(max_0 - min_0)
        # print( feature_noise[:,0] )
        # pdb.set_trace()
            feature_noise[:,1] = (feature_noise[:,1] - min_1)/(max_1 - min_1)
        feature_normal = np.array(feature_normal)
        feature_normal[:,0] = (feature_normal[:,0] - min_0)/(max_0 - min_0)
        feature_normal[:,1] = (feature_normal[:,1] - min_1)/(max_1 - min_1)
        feature_normal_center = [np.mean(feature_normal[:,0]), np.mean(feature_normal[:,1])]
        # print(feature_normal_center)
        feature_noise_dis = 0
        if len(feature_noise) > 0:
            for i in range(len(feature_noise)):
                feature_noise_dis += (feature_noise[i][0] - feature_normal_center[0])**2 + (feature_noise[i][1] - feature_normal_center[1])**2
        # print(feature_noise_dis)
        # pdb.set_trace()

        center_dis += feature_noise_dis
        # print(feature_noise)
        # print(mask)
        
        if len(feature_noise) > 0:
            x = feature_noise[:, 1]
            y = feature_noise[:, 0]
            plt.plot(x, y, "^", color="black")
        x = feature_normal[:, 1]
        y = feature_normal[:, 0]
        plt.plot(x, y, '.', color=color[j])
        #plt.legend(label, loc="upper right")     #如果写在plot上面，则标签内容不能显示完整
    plt.title("epoch={}, C_H = {}, center_dis ={}".format(str(epoch), str(int(score)), str(center_dis)))
 
    # plt.savefig('{}/{}.jpg'.format(save_path,epoch+1))
    plt.savefig(save_path + '_' + str(epoch) + '_' + str(batch_i) + '_acc_' + str(int(acc*100)) + '.jpg')

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 10, drop_rate = 0, t_adj = 0.5):
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
        self.agbn3 = AdapGBN(512, 512)
        self.relu = nn.LeakyReLU(0.4)
        self.t = t_adj
        self.BN = nn.BatchNorm1d(fc_in_dim)

    def backward_agbnfer(self, lr):
        self.agbn1.backward_agbn(lr)
        self.agbn2.backward_agbn(lr)
        

    def forward(self, x, phase, threshold):
        x = self.features(x)
        # print(x.shape)
        # pdb.set_trace()
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)  

        cs1 = cosine_similarity(x.cpu().detach())
        x = self.agbn1(x, cs1, phase, threshold)
            
        x = self.relu(x)
        cs2 = cosine_similarity(x.cpu().detach())
        x = self.agbn2(x, cs2, phase, threshold)
        x2d = self.fc2d(x)
        out2d = self.fc2dcls(x2d)
        return out2d, x2d
        
def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
        # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # init_weight_fn = get_condconv_initializer(
            # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        # init_weight_fn(m.weight)
        # if m.bias is not None:
            # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
        
def run_training(t_adj = 0.5):
    
    args = parse_args()
    imagenet_pretrained = True

    ## model_choose
    # modelfer = EfficientNet(num_classes = 11, t_adj = 0.5)
    modelfer = Res18Feature(pretrained = imagenet_pretrained, drop_rate = args.drop_rate, t_adj = t_adj) 
    if not imagenet_pretrained:
         for m in modelfer.modules():
            initialize_weight_goog(m)
    ## model_choose
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']
        # for param_tensor in pretrained_state_dict: # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor,'\t',pretrained_state_dict[param_tensor].size())
        # pdb.set_trace()
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
        modelfer.load_state_dict(model_state_dict, strict = False)   
        
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    
    datapathlist_train = [cfg.minst_path]
    labelpathlist_train = [cfg.minst_draw2d_withnoise10]
    train_dataset = FERDataSet(datapathlist_train, labelpathlist_train, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            #    sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 
    datapathlist_test = [cfg.minst_path]
    labelpathlist_test = [cfg.minst_draw2d_withnoise10]                                          
    val_dataset = FERDataSet(datapathlist_test, labelpathlist_test, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            #    sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    ## model_choose
    # params = res18.parameters()
    params = modelfer.parameters()
    ## model_choose

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    ## model_choose
    modelfer = modelfer.cuda()
    ## model_choose
    criterion = torch.nn.CrossEntropyLoss()
    max_acc = 0
    acc = 0
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        ## model_choose
        # res18.train()
        modelfer.train()
        ## model_choose
        for batch_i, (imgs, targets, path, indexes) in enumerate(train_loader):
            modelfer.train()
            batch_sz = imgs.size(0) 
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            ## model_choose
            # attention_weights, outputs = res18(imgs)
            outputs, feature = modelfer(imgs, "train", 0.5)
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            loss.backward()
            # optimizer_centloss.zero_grad()
            # loss.backward()
            # for param in center_loss.parameters():
            #     param.grad.data *= (1./alpha)
            # optimizer_centloss.step()
            optimizer.step()
            modelfer.backward_agbnfer(0.001)
                # if cate == 2:
                    # modelfer.backward_agbnfer(0.01)
                
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            acc = correct_num.float() / float(1000)
            print(batch_i)
            decet_withindex(index_noise, acc, feature,targets,i,batch_i,'AGBNminst_')
                #decet(correct_num.cpu().numpy()/args.batch_size, feature,predicts,i,batch_i,'/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/feature2dAGBN/train_preds_' + catedict[cate])

        if i%1 == 0:
            scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

     
            
if __name__ == "__main__":
    run_training(0.5)
