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
# from center_loss import CenterLoss
# center_loss = CenterLoss(num_classes=8, feat_dim=512, use_gpu=True)
# optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)
# alpha = 0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import timm
batchsize_global = 256

# from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator

# class DataLoaderX(DataLoader):

#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/home/dyt/FER_workspace/FERdataset/RAFDB/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default="/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/AGBN_affect_epoch8_acc0.6138.pth",
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=55, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.07, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=batchsize_global, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 8, drop_rate = 0):
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
        #lf.agbn3 = AdapGBN(512, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.t = 0.5
        self.BN = nn.BatchNorm1d(fc_in_dim)
    
    def backward_agbnfer(self, lr):
        self.agbn1.backward_agbn(lr)
        self.agbn2.backward_agbn(lr)
        #self.agbn3.backward_agbn(lr)


    def forward(self, x, phase, threshold):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        # out = self.fc(x)
        # print(out)
        # pdb.set_trace()
        # return out, x
        # print(x)
        cs1 = cosine_similarity(x.cpu().detach())
        x = self.agbn1(x, cs1, phase, threshold)
        x = self.relu(x)

        cs2 = cosine_similarity(x.cpu().detach())
        x = self.agbn2(x, cs2, phase, threshold)
        # x = self.relu(x)

        # cs3 = cosine_similarity(x.cpu().detach())
        # x = self.agbn2(x, cs3, phase, threshold)

        out = self.fc(x)
        # # print(out)
        # # pdb.set_trace()
        return out, x
        
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
        
def run_training(fold, threshold):
    
    args = parse_args()
    imagenet_pretrained = True

    ## model_choose
    # modelfer = EfficientNet(num_classes = 11, t_adj = 0.5)
    modelfer = Res18Feature(pretrained = imagenet_pretrained, drop_rate = args.drop_rate) 
    ## model_choose

    # device_ids = [0,1]
    # modelfer = torch.nn.DataParallel(modelfer, device_ids=device_ids)

    ## model_choose
    # if not imagenet_pretrained:
    #      for m in res18.modules():
    #         initialize_weight_goog(m)
    if not imagenet_pretrained:
         for m in modelfer.modules():
            initialize_weight_goog(m)
    ## model_choose
            
    # if args.pretrained:
    if False:
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
    print(cfg.affect_noise20_label_path)
    datapathlist_train = [cfg.affect_path ]
    labelpathlist_train = [cfg.affect_noise20_label_path ]
    train_dataset = FERDataSet(datapathlist_train, labelpathlist_train, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            #    sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 
    datapathlist_test = [cfg.affect_path ]
    labelpathlist_test = [cfg.affect_label_path  ]                                          
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
    # res18 = res18.cuda()
    modelfer = modelfer.cuda()
    ## model_choose
    weight_CE_Affectnet = torch.FloatTensor([1.79, 1, 5.27, 9.53, 21.07, 35.34, 5.40, 35.84])
    # weight_CE_Affectnet = torch.FloatTensor([1.79, 1, 5.27, 9.53, 21.07, 35.34, 5.40, 35.84])
    criterion = torch.nn.CrossEntropyLoss(weight = weight_CE_Affectnet.cuda())
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    max_acc = 0
    fvalid = open('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/Affectcrop_AGBN_testn20' + str(batchsize_global) + '_' + str(fold) + '_threshold' + str(threshold) + '.txt', 'a+')
    ftrain = open('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/Affectcrop_AGBN_trainn20' + str(batchsize_global) + '_' + str(fold) + '_threshold' + str(threshold) + '.txt', 'a+')
    ftrain_log = open('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/Affectcrop_AGBN_trainlogn20' + str(batchsize_global) + '_' + str(fold) + '_threshold' + str(threshold) + '.txt', 'a+')
    fvalid_log = open('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/Affectcrop_AGBN_testlogn20' + str(batchsize_global) + '_' + str(fold) + '_threshold' + str(threshold) + '.txt', 'a+')
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
            tops = int(batch_sz* beta)
            optimizer.zero_grad()
            imgs = imgs.cuda()
            ## model_choose
            # attention_weights, outputs = res18(imgs)

            outputs, feature = modelfer(imgs, "train", threshold) 
            score, _ = torch.max(outputs, 1)
            # print(score)
            # # print(targets)
            # print(outputs)
            # pdb.set_trace()
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            loss.backward()
            # optimizer_centloss.zero_grad()
            # loss.backward()
            # for param in center_loss.parameters():
            #     param.grad.data *= (1./alpha)
            # optimizer_centloss.step()
            # for name, parms in modelfer.named_parameters():	
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad)
            #     if name == 'gc1.adjwithgrad':
            #         print(parms.grad)
            #         print(parms.grad.shape)
            
            optimizer.step()
            # modelfer.backward_agbnfer(0.001)
            
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            # print(predicts.cpu().numpy())
            # print(targets.cpu().numpy())
            # for (demo_t, demo_p) in zip(targets.cpu().numpy(), predicts.cpu().numpy()):
            #     print(demo_t, demo_p)
            #     break
            # pdb.set_trace()
            ftrain_log.write("epoch: " + str(i) + " batch: " + str(batch_i) + "\n")
            for (demo_t, demo_p) in zip(targets.cpu().numpy(), predicts.cpu().numpy()):
                ftrain_log.write("traget: " + str(demo_t) + " preds: " + str(demo_p) + "\n")
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            # print("train batch_id : ", batch_i, "batch correct num: ", correct_num)

        if True:
            scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        ftrain.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (i, acc, running_loss))
        if acc > 0.95:
            torch.save({'iter': i,
                            'model_state_dict': modelfer.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/experiments_save', "AGBN_affect_epoch"+str(i)+"_acc"+str(acc.cpu().numpy())+ ".pth"))
            print('Model saved.')
            break
        # pdb.set_trace()
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            ## model_choose
            # res18.eval()
            modelfer.eval()
            ## model_choose
            for batch_i, (imgs, targets, _, _) in enumerate(val_loader):
                ## model_choose
                # _, outputs = res18(imgs.cuda())
                outputs,feature = modelfer(imgs.cuda(), "val", threshold)
                # print(targets)
                # print(feature)
                # pdb.set_trace()
                ## model_choose
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(outputs, 1)
                fvalid_log.write("epoch: " + str(i) + " batch: " + str(batch_i) + "\n")
                for (demo_t, demo_p) in zip(targets.cpu().numpy(), predicts.cpu().numpy()):
                    fvalid_log.write("traget: " + str(demo_t) + " preds: " + str(demo_p) + "\n")
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                # print("valid batch_id : ", batch_i, "batch correct num: ", correct_num.sum().cpu())
                
            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))
            fvalid.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f \n" % (i, acc, running_loss))
            if max_acc < acc:
                max_acc = acc
            print("max acc: ", max_acc)
            if max_acc > 0.57:
                torch.save({'iter': i,
                            'model_state_dict': modelfer.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('/home/dyt/FER_workspace/Self-Cure-Network-master/src/cvpr2022/AffectNetResults/', "AGBN_affectn10_epoch"+str(i)+"_acc"+str(acc)+ ".pth"))
                print('Model saved.')
                break
     
            
if __name__ == "__main__":
    for fold in range(10):
        run_training(fold, 0.5)
