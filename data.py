
import os
import sys
import glob
import h5py
import random
import numpy as np
from torch.utils.data import Dataset
import torch



def load_data(partition, level):
    DATA_DIR = "/172.31.222.52_data/pufan_li"
    all_data = []
    all_label = []
    all_pid = []#hdf5_data
    cnt = 0
    for _ in sorted(glob.glob(os.path.join(DATA_DIR, 'smallshapenet_seg_h5', str(level), 'ply_data_%s*.h5'%partition)), key=os.path.getmtime):
        h5_name = os.path.join(DATA_DIR, 'smallshapenet_seg_h5', str(level), 'ply_data_%s%s.h5'%(partition,str(cnt)))
        cnt += 1
        print(h5_name)
        f = h5py.File(h5_name)
        #for q in f.keys():
         #  print(f[q], q, f[q].name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        pid = f['pid'][:].astype('int64')
        #print(label.shape)
        #print(pid.shape)
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_pid.append(pid)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_pid = np.concatenate(all_pid, axis=0)
    #print(all_data.shape)
    #print(all_label.shape)
    #print(all_pid.shape)
    return all_data, all_label, all_pid
    
def load_data_noise(partition, std_min, std_max, rotation = False):
    DATA_DIR = "/172.31.222.52_data/pufan_li"
    all_data = []
    all_label = []
    all_pid = []#hdf5_data
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, 'smallshapenet_seg_h5', '0', 'ply_data_%s*.h5'%partition)), key=os.path.getmtime):
        f = h5py.File(h5_name)
        print(h5_name)
        #for q in f.keys():
         #  print(f[q], q, f[q].name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        pid = f['pid'][:].astype('int64')
        #print(data.shape)
        #print(label.shape)
        #print(pid.shape)
        f.close()
        for i in range(data.shape[0]):
            noise = random.uniform(std_min, std_max)
            noi = np.random.rand(data.shape[1], data.shape[2]) * noise
            data[i, :, :] += noi
        all_data.append(data)
        all_label.append(label)
        all_pid.append(pid)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_pid = np.concatenate(all_pid, axis=0)
    #print(all_data.shape)
    #print(all_label.shape)
    #print(all_pid.shape)
    return all_data, all_label, all_pid


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train', level = 0):
        print(partition + str(level))
        self.data, self.label, self.pid = load_data(partition, level)
        self.num_points = num_points
        self.partition = partition        
        print(self.data.shape[0])

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        pid = self.pid[item]
        return pointcloud, label, pid

    def __len__(self):
        return self.data.shape[0]

class double_ShapeNet(Dataset):
    def __init__(self, num_points, partition = 'train', level = 0):
        print('double ' + partition + ' ' + str(level))
        self.data, self.label, self.pid = load_data(partition, level)
        self.data1, self.label1, self.pid1 = load_data(partition, 0)
        self.num_points = num_points
        self.partition = partition
    
    def __getitem__(self, item):
        data = self.data[item][:self.num_points]
        label = self.label[item]
        pid = self.pid[item]
        gt_pc = self.data1[item][:self.num_points]
        gt_label = self.label1[item]
        gt_pid = self.pid1[item]
        return data, label, pid, gt_pc, gt_label, gt_pid

    def __len__(self):
        return self.data.shape[0]

class random_noise_shapenet(Dataset):
    def __init__(self, num_points, partition, noise_min, noise_max, rotation = False):
        print('{:}, {:}, {:}'.format(partition, noise_min, noise_max))
        self.data, self.label, self.pid = load_data(partition, 0)
        self.data1, self.label1, self.pid1 = load_data_noise(partition, noise_min, noise_max, rotation)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        data = self.data1[item][:self.num_points]
        label = self.label1[item]
        pid = self.pid1[item]
        gt_pc = self.data[item][:self.num_points]
        gt_label = self.label[item]
        gt_pid = self.pid[item]
        return data, label, pid, gt_pc, gt_label, gt_pid

    def __len__(self):
        return self.data.shape[0]
        
if __name__ == '__main__':
    a = ShapeNet(2048,'test')
    '''train = random_noise_shapenet(2048, 'train', 0.05, 0.3)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, 'denoise_pc', 'random_noise')
    cnt = 0
    for data, _, _, gt_pc, _, _ in train:
        save_file = os.path.join(file_dir, "%s.xyz"%(cnt+1))
        np.savetxt(save_file, data, '%.8f')
        save_file = os.path.join(file_dir, "gt%s.xyz"%(cnt+1))
        np.savetxt(save_file, gt_pc, '%.8f')
        cnt += 1
        if cnt == 10:
            break'''
    '''train = ShapeNet(2048)
    print("train finish")
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, 'pc0.03.xyz')
    print(file_dir)
    for data, label, pid in train:
        np.savetxt(file_dir, data, '%.8f')
        print(data[0, :])
        break'''
