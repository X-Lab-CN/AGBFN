import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
from collections import Counter
import pdb
## rafdb
# 1: Surprise 2: Fear 3: Disgust 4: Happiness 5: Sadness 6: Anger 7: Neutral
##ferplus
# 1:neutral 2:happiness 3:surprise 4:sadness 5:anger 6:disgust 7:fear 8:contempt 9:unknown
class FERDataSet(data.Dataset):
    def __init__(self, data_path_list, label_path_list, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path_list[0]
        self.label_path = label_path_list[0]
        
        self.file_paths = []
        self.label = []
        with open(self.label_path, 'r') as fin:
            tmp = fin.readlines()
            for demo in tmp:
                tmpp = demo.split(' ')
                if phase == 'train':
                    if tmpp[0] == 'train' or tmpp[0] == 'train':
                        self.file_paths.append(self.data_path + tmpp[1])
                        # print(self.data_path)
                        if "FER2013plus" in self.data_path:
                            self.label.append(int(tmpp[2])) #FER2013plus
                        # pdb.set_trace()
                        else:
                            self.label.append(int(tmpp[2])) #RAFDB
                        #  #affect ferplus
                else:
                    if tmpp[0] == phase:
                        #affectnet
                        # if int(tmpp[2]) == 7:
                        #     continue
                        self.file_paths.append(self.data_path + tmpp[1])
                        if "FER2013plus" in self.data_path:
                            self.label.append(int(tmpp[2])) #FER2013plus
                        else:
                            self.label.append(int(tmpp[2])) #RAFDB
                        # self.label.append(int(tmpp[2]) - 1) #rafdb
                        # self.label.append(int(tmpp[2])) #affect ferplus
        ## use for ferplus and rafdb
        # if phase == 'train':
        #     self.data_path1 = data_path_list[1]
        #     self.label_path1 = label_path_list[1]
        #     with open(self.label_path1, 'r') as fin:
        #         tmp = fin.readlines()
        #         for demo in tmp:
        #             tmpp = demo.split(' ')
        #             self.file_paths.append(self.data_path1 + tmpp[1])
        #             self.label.append(int(tmpp[2]) - 1)
        #             # self.label.append(int(tmpp[2])) #affect ferplus
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
        
        c = Counter(self.label)
        print(dict(c))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # print(path)
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, path, idx
    
    def get_labels(self):
        return self.label

