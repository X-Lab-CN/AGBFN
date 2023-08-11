import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy as np
import math
import pdb
import random

class AdapGBN(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(AdapGBN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_weight = Parameter(torch.Tensor(in_features, out_features))
        #self.adjwithgrad = Parameter(torch.Tensor(batchsize, batchsize))
        # self.threshold_possion = torch.Tensor(batchsize, batchsize)
        self.alpha_agbn = Parameter(torch.Tensor(1))
        self.alpha_agbn.data.fill_(1.4)
        self.beta_agbn = Parameter(torch.Tensor(1))
        self.beta_agbn.data.fill_(0)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.normalize_weight.size(1))
        self.normalize_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def gen_A_Ber(self, _adj, t):
        # print(_adj.shape)
        beta = 2
        alpha = 1/(1 + np.exp(-1*beta*(_adj - t)))
        # print(alpha)
        # print(alpha.shape)
        # pdb.set_trace()
        ber_matrix = np.zeros((_adj.shape[0], _adj.shape[1]))
        for i in range(ber_matrix.shape[0]):
            for j in range(ber_matrix.shape[1]):
                ber_matrix[i][j] = random.random()
        connect_matrix = alpha > ber_matrix
        _adj[connect_matrix == True] = 1
        _adj[connect_matrix == False] = 0
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + np.identity(_adj.shape[0], np.int)
        print(_adj)
        pdb.set_trace()
        return _adj
    
    def gen_A_possion(self, cs, phase, threshold):
        S = 1400
        lambda_possion = self.alpha_agbn.data.cpu().numpy()*cs + self.beta_agbn.data.cpu().numpy()
        lambda_possion[lambda_possion < 0] = 0
        self.threshold_possion = 1- np.exp(-1*lambda_possion)
        self.cs = cs
        sample_init = 1/(1+1/np.sqrt(np.exp(1)))
        sample = 0
        for i in range(S):
            sample += np.random.rand(1)
        sample = sample / S 
        if phase == "val":
            sample = threshold
        # sample = 0.5
        # self.threshold_possion = cs
        cs[sample < self.threshold_possion] = 1
        cs[sample >= self.threshold_possion] = 0
        _adj = torch.from_numpy(cs).float().cuda()
        _adj = _adj * 0.5/ (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + torch.from_numpy(np.identity(_adj.shape[0], np.int)).float().cuda()

        return _adj


    def forward(self, input, cs, phase, threshold):
        support = torch.matmul(input, self.normalize_weight)
        adj_A = self.gen_adj(self.gen_A_possion(cs, phase, threshold)).cuda()
        self.temp_adjwithgrad = torch.Tensor(adj_A.shape[0], adj_A.shape[0]).requires_grad_()
        self.temp_adjwithgrad.data = adj_A
        output = torch.matmul(self.temp_adjwithgrad, support)
        self.temp_adjwithgrad.register_hook(lambda grad: grad)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def backward_agbn(self, lr):
        grad_adj_F = self.temp_adjwithgrad.grad.data.cpu().numpy()
        # try:
        # grad_adj_F = self.adjwithgrad.grad.data.cpu().numpy()
        # print(self.adjwithgrad.grad.data.cpu().numpy())
        # print(self.temp_adjwithgrad.grad)
        # pdb.set_trace()
        # except:
        #     grad_adj_F = np.zeros((self.threshold_possion.shape[0], self.threshold_possion.shape[0]))
        # grad_alpha_adj = 1/grad_adj_F.shape[0]/grad_adj_F.shape[0]/self.alpha_agbn * (self.threshold_possion @ grad_adj_F).sum()
        # grad_beta_adj = 1/grad_adj_F.shape[0]/grad_adj_F.shape[0]*((self.threshold_possion * (1 - self.threshold_possion)) @ grad_adj_F).sum()
        # print(grad_alpha_adj)
        grad_alpha_adj = 1/grad_adj_F.shape[0]/grad_adj_F.shape[0]/(self.cs @ grad_adj_F).sum()
        grad_beta_adj = 1/grad_adj_F.shape[0]/grad_adj_F.shape[0]*(grad_adj_F).sum()
        self.alpha_agbn.data -= lr*grad_alpha_adj
        self.beta_agbn.data -= lr*grad_beta_adj
        # pdb.set_trace()


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'