import numpy as np
import os
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def get_adj_matrix(x, k, pp = False):
    #print(x[0,0:10,:])
    x = x.permute(0, 2, 1)
    B, M, N = x.shape
    inner = 2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    a = xx - inner + xx.transpose(2, 1) #两两距离
   # print(a[0,0,0:10])
    #print(a[0,0:10,0:10])
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    b = a.topk(k, dim = -1,largest = False)[0]
    for_var = a.topk(int(k * 2.0), dim = -1,largest = False)[0]
    var = torch.var(for_var, dim = (-1, -2))
    var = torch.unsqueeze(var, dim = -1)
    var = torch.unsqueeze(var, dim = -1)
    var = var.repeat(1, N, N)
    '''delta = (b.sum(dim = -1)).sum(dim = -1)
    delta = delta / (N * k)
    delta = delta.unsqueeze(-1)
    delta = delta.repeat(1, N)
    delta = delta.unsqueeze(-1)
    delta = delta.repeat(1, 1, N)'''
    
    b = b.max(dim = -1, keepdim = True)[0]
    c = torch.where(a<=b, torch.exp(-a / var), torch.zeros_like(a))
    return c

def syn_loss(pc, gt_pc, pred, gold, k_nei):
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    B, N, _ = pc.shape
    loss_1 = cal_loss(pred, gold)
    loss_2, _ = chamfer_distance(pc, gt_pc)
    A = get_adj_matrix(gt_pc, k = k_nei)
    D = A.sum(dim = -1)
    D = 1 / torch.sqrt(D)
    L = D.unsqueeze(-1) * A
    L = D.unsqueeze(-2) * L
    test = L.sum(dim=-1)
    I = torch.eye(N).repeat(B, 1, 1).to(device)
    L = I - L
    ans = torch.bmm(torch.bmm(pc.permute(0, 2, 1), L), pc)
    bb = torch.diagonal(ans, dim1 = -1, dim2 = -2)
    cc = torch.sum(bb, dim = -1)
    loss_3 = cc.mean()
    #loss_3 = (ans[:, 0, 0] + ans[:, 1, 1] + ans[:, 2, 2]).mean()
    return loss_1, loss_2, loss_3

def loss_without_L2(pc, gt_pc, pred, gold, k_nei):
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    B, N, _ = pc.shape
    loss_1 = cal_loss(pred, gold)
    A = get_adj_matrix(gt_pc, k = k_nei)
    D = A.sum(dim = -1)
    D = 1 / torch.sqrt(D)
    L = D.unsqueeze(-1) * A
    L = D.unsqueeze(-2) * L
    test = L.sum(dim=-1)
    I = torch.eye(N).repeat(B, 1, 1).to(device)
    L = I - L
    ans = torch.bmm(torch.bmm(pc.permute(0, 2, 1), L), pc)
    bb = torch.diagonal(ans, dim1 = -1, dim2 = -2)
    cc = torch.sum(bb, dim = -1)
    loss_3 = cc.mean()
    #loss_3 = (ans[:, 0, 0] + ans[:, 1, 1] + ans[:, 2, 2]).mean()
    return loss_1, loss_3

def loss_without_L3(pc, gt_pc, pred, gold, k_nei):
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    B, N, _ = pc.shape
    loss_1 = cal_loss(pred, gold)
    loss_2, _ = chamfer_distance(pc, gt_pc)
    return loss_1, loss_2

def batch_quat_to_rotmat(q, out=None): #q:Batchsize * 4, out: B * 3 * 3, 由四元数返回一个旋转矩阵，用哈密尔顿转换法

    batchsize = q.size(0)

    if out is None:
        out = torch.FloatTensor(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out

class IOStream():
    def __init__(self, path):
        dir = os.path.abspath(path)
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.f = open(path + "/run.log", 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()