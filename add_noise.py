import argparse
import os
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from data import *
import time

def add_noise(d, partition, noise):
    dir_name = "/172.31.222.52_data/pufan_li/smallshapenet_seg_h5"
    if os.path.exists(dir_name) == 0:
        os.mkdir(dir_name)
    dir_name = os.path.join(dir_name, str(noise))
    if os.path.exists(dir_name) == 0:
        os.mkdir(dir_name)
    cnt = 0
    now = 0
    all_data = []
    all_label = []
    all_pid = []
    for data, label, pid in d:
        noi = np.random.rand(data.shape[0], data.shape[1]) * noise
        new_d = data + noi
        all_data.append(new_d[np.newaxis, :])
        all_label.append(label[np.newaxis, :])
        all_pid.append(pid[np.newaxis, :])
        now += 1
        if now == 2048:
            time.sleep(1)
            now = 0
            all_data = np.concatenate(all_data, axis=0)
            all_label = np.concatenate(all_label, axis=0)
            all_pid = np.concatenate(all_pid, axis=0)
            file_name = os.path.join(dir_name, 'ply_data_%s%d.h5'%(partition,cnt))
            F = h5py.File(file_name, "w")
            print(file_name)
            print(all_data.shape)
            F.create_dataset('pointcloud',data=np.array(["data".encode(),"label".encode(), "pid".encode()]))
            F.create_dataset("data", dtype='float32', data = all_data)
            F.create_dataset("label", dtype = 'int8', data = all_label)
            F.create_dataset("pid", dtype = 'int8', data = all_pid)
            F.close()
            all_data = []
            all_label = []
            all_pid = []
            cnt = cnt + 1
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_pid = np.concatenate(all_pid, axis=0)
    file_name = os.path.join(dir_name, 'ply_data_%s%d.h5'%(partition,cnt))
    F = h5py.File(file_name, "w")
    print(file_name)
    print(all_data.shape)
    F.create_dataset('pointcloud',data=np.array(["data".encode(),"label".encode(), "pid".encode()]))
    F.create_dataset("data", dtype='float32', data = all_data)
    F.create_dataset("label", dtype = 'int8', data = all_label)
    F.create_dataset("pid", dtype = 'int8', data = all_pid)

if __name__ == '__main__':
    noise = 0.05
    train = ShapeNet(2048, 'train')
    add_noise(train, 'train', noise)
    test = ShapeNet(2048, 'test')
    add_noise(test, 'test', noise)
    val = ShapeNet(2048, 'val')
    add_noise(val, 'val', noise)
    #test = ShapeNet(2048, 'test')
    #val = ShapeNet(2048, 'val')
    
        