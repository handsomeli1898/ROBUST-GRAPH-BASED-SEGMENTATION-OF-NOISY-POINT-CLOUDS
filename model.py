import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class DGCNN_partseg_modify(nn.Module):
    def __init__(self, args):
        super(DGCNN_partseg_modify, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, args.seg_number, kernel_size=1, bias=False)

        bn12 = nn.BatchNorm1d(256)
        self.conv12 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size = 1, bias = False),
                                    bn12,
                                    nn.LeakyReLU(negative_slope=0.2))
        bn13 = nn.BatchNorm1d(256)
        self.conv13 = nn.Sequential(nn.Conv1d(256, 256, kernel_size = 1, bias = False),
                                    bn13,
                                    nn.LeakyReLU(negative_slope=0.2))
        bn14 = nn.BatchNorm1d(128)
        self.conv14 = nn.Sequential(nn.Conv1d(256, 128, kernel_size = 1, bias = False),
                                    bn14,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv15 = nn.Sequential(nn.Conv1d(128, 3, kernel_size = 1, bias = False), nn.Tanh())
        
    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        y1 = self.conv12(x)                      # (B, 1088+64*3, N) -> (B, 256, N)
        y2 = self.conv13(y1)                      # (B, 256, N) -> (B, 256, N)
        y3 = self.conv14(y2)                      # (B, 256, N) -> (B, 128, N)
        y = self.conv15(y3)                      # (B, 128, N) -> (B, 3, N)

        z = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        z = self.dp1(z)
        z = self.conv9(z + y1)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        z = self.dp2(z)
        z = self.conv10(z + y2)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        z = self.conv11(z + y3)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)

        return z, y

class gcnconv(nn.Module):
    def __init__(self, in_channel, out_channel, bias = False):
        super(gcnconv, self).__init__()
        self.in_feature = in_channel
        self.out_feature = out_channel
        self.W = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        std = 1./math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std,std)
        self.use_bias = bias
        if bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(out_channel))
            self.bias.data.uniform_(-std,std)

    def forward(self, x, A):
        y = torch.matmul(x, self.W)
        y = torch.matmul(A, y)
        if self.use_bias == True:
            y = y + self.bias
        return y

class gcnmodule(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(gcnmodule, self).__init__()
        self.gcn1 = gcnconv(in_channel, hidden_channel)
        self.gcn2 = gcnconv(hidden_channel, out_channel)
        self.l_r_1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, A): #x:(B,F,N)
        x = x.permute(0, 2, 1)
        h = self.l_r_1(self.gcn1(x, A))
        y = self.gcn2(h, A)
        y = y.permute(0, 2, 1)
        return y


def get_adj_matrix(x, k, pp = False):
    B, M, N = x.shape
    inner = 2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    a = xx - inner + xx.transpose(2, 1) #两两距离
   # print(a[0,0,0:10])
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    b = a.topk(k, dim = -1,largest = False)[0]
    for_var = a.topk(int(k * 2.0), dim = -1,largest = False)[0]
    if pp == False or 1 == 1:
        var = torch.var(for_var, dim = (-1, -2))
        var = torch.unsqueeze(var, dim = -1)
        var = torch.unsqueeze(var, dim = -1)
        var = var.repeat(1, N, N)
    else:
        var = torch.var(for_var)
    b = b.max(dim = -1, keepdim = True)[0]
    c = torch.where(a<=b, torch.exp(-a / var), torch.zeros_like(a))
    I = torch.eye(N).repeat(B, 1, 1).to(device)
    D = c.sum(dim = -1)
    D = 1 / torch.sqrt(D)
    g = D.unsqueeze(-1) * c
    g = D.unsqueeze(-2) * g
    return g

class GCN_partseg(nn.Module):
    def __init__(self, args):
        super(GCN_partseg, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.gcn1 = gcnmodule(in_channel = 3, hidden_channel = 6, out_channel = 6)
        self.gcn2 = gcnmodule(in_channel = 64, hidden_channel = 64, out_channel = 64)
        self.gcn3 = gcnmodule(in_channel = 64, hidden_channel = 64, out_channel = 64)
        self.gcn1_ac = nn.LeakyReLU(negative_slope = 0.2)
        self.gcn2_ac = nn.LeakyReLU(negative_slope = 0.2)
        self.gcn3_ac = nn.LeakyReLU(negative_slope = 0.2)

        self.bn0_0 = nn.BatchNorm1d(6)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)

        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size = 1, bias = False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope = 0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, args.seg_number, kernel_size=1, bias=False)
        
    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
       #print(x[0,:,2002])
        A = get_adj_matrix(x, self.k, self.args.train)
        x = self.gcn1(x, A)
        x = self.bn0_0(x)
        x = self.gcn1_ac(x)
        x = self.conv1(x)
        x1 = self.conv2(x)
        A = get_adj_matrix(x1, self.k, self.args.train)
        x = self.gcn2(x1, A)
        x = self.bn0_1(x)
        x = self.gcn2_ac(x)
        x = self.conv3(x)
        x2 = self.conv4(x)

        A = get_adj_matrix(x2, self.k, self.args.train)
        x = self.gcn3(x2, A)
        x = self.bn0_2(x)
        x = self.gcn3_ac(x)
        x3 = self.conv5(x)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        z = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        z = self.dp1(z)
        z = self.conv9(z)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        z = self.dp2(z)
        z = self.conv10(z)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        z = self.conv11(z)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        
        return z

class GCN_partseg_modify(nn.Module):
    def __init__(self, args):
        super(GCN_partseg_modify, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.gcn1 = gcnmodule(in_channel = 3, hidden_channel = 6, out_channel = 6)
        self.gcn2 = gcnmodule(in_channel = 64, hidden_channel = 64, out_channel = 64)
        self.gcn3 = gcnmodule(in_channel = 64, hidden_channel = 64, out_channel = 64)
        self.gcn1_ac = nn.LeakyReLU(negative_slope = 0.2)
        self.gcn2_ac = nn.LeakyReLU(negative_slope = 0.2)
        self.gcn3_ac = nn.LeakyReLU(negative_slope = 0.2)

        self.bn0_0 = nn.BatchNorm1d(6)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)

        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size = 1, bias = False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope = 0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1, bias = False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope = 0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, args.seg_number, kernel_size=1, bias=False)

        bn12 = nn.BatchNorm1d(256)
        self.conv12 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size = 1, bias = False),
                                    bn12,
                                    nn.LeakyReLU(negative_slope=0.2))
        bn13 = nn.BatchNorm1d(256)
        self.conv13 = nn.Sequential(nn.Conv1d(256, 256, kernel_size = 1, bias = False),
                                    bn13,
                                    nn.LeakyReLU(negative_slope=0.2))
        bn14 = nn.BatchNorm1d(128)
        self.conv14 = nn.Sequential(nn.Conv1d(256, 128, kernel_size = 1, bias = False),
                                    bn14,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv15 = nn.Sequential(nn.Conv1d(128, 3, kernel_size = 1, bias = False), nn.Tanh())
        
    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        A = get_adj_matrix(x, self.k, self.args.train)
        x = self.gcn1(x, A)
        x = self.bn0_0(x)
        x = self.gcn1_ac(x)
        x = self.conv1(x)
        x1 = self.conv2(x)

        A = get_adj_matrix(x1, self.k, self.args.train)
        x = self.gcn2(x1, A)
        x = self.bn0_1(x)
        x = self.gcn2_ac(x)
        x = self.conv3(x)
        x2 = self.conv4(x)

        A = get_adj_matrix(x2, self.k, self.args.train)
        x = self.gcn3(x2, A)
        x = self.bn0_2(x)
        x = self.gcn3_ac(x)
        x3 = self.conv5(x)
        
        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        y1 = self.conv12(x)                      # (B, 1088+64*3, N) -> (B, 256, N)
        y2 = self.conv13(y1)                      # (B, 256, N) -> (B, 256, N)
        y3 = self.conv14(y2)                      # (B, 256, N) -> (B, 128, N)
        y = self.conv15(y3)                      # (B, 128, N) -> (B, 3, N)
        
        z = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        z = self.dp1(z)
        z = self.conv9(z + y1)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        z = self.dp2(z)
        z = self.conv10(z + y2)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        z = self.conv11(z + y3)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        
        return z, y
