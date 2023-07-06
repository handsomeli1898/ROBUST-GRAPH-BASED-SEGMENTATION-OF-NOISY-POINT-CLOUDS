from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import *
from model import *
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if class_choice is None:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious
def show_denoise(args, io):
    print("show")
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.method == 'dgcnn':
        model = DGCNN_partseg_modify(args).to(device)
    else:
        raise Exception("Not implemented")
        
    model = nn.DataParallel(model)
    
    abs_file = os.path.dirname(os.path.abspath(__file__)) 
    print(os.path.join(abs_file, args.model_path))
    model.load_state_dict(torch.load(os.path.join(abs_file, args.model_path)))
    print(args.model_path)
    save_file = os.path.join(abs_file, 'denoise_pc', args.model_name)
    if os.path.exists(save_file) == 0:
        os.mkdir(save_file)
    save_file = os.path.join(save_file, str(args.noise_level))
    if os.path.exists(save_file) == 0:
        os.mkdir(save_file)
    model.train()
    test_loader =  DataLoader(double_ShapeNet(num_points=args.num_points, partition='test', level = args.noise_level), 
                              num_workers=2, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    cnt = 0
    criterion = syn_loss
    with torch.no_grad():
        for data, label, seg, gt_pc, _, _ in test_loader:
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg, gt_pc = data.to(device), label_one_hot.to(device), seg.to(device), gt_pc.to(device)
            for j in range(args.test_batch_size):
                save_file_1 = os.path.join(save_file, "origin%s.xyz"%(j+1))
                np.savetxt(save_file_1, data[j, :, :].cpu(), "%.8f")
                save_file_1 = os.path.join(save_file, "gt%s.xyz"%(j+1))
                np.savetxt(save_file_1, gt_pc[j, :, :].cpu(), "%.8f")
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred, denoise_pc = model(data, label_one_hot)
            cnt += 1
            cd_loss, abc = chamfer_distance(denoise_pc.permute(0, 2, 1).contiguous(), gt_pc)
            denoise_pc = denoise_pc.permute(0,2,1)
            for j in range(args.test_batch_size):
                save_file_1 = os.path.join(save_file, "%s.xyz"%(j+1))
                np.savetxt(save_file_1, denoise_pc[j, :, :].detach().cpu().numpy(), "%.8f")
            break
            
   

def test(args, io):
    print('test')
    test_loader = DataLoader(double_ShapeNet(num_points=args.num_points, partition='test', level = args.noise_level), 
                              num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.method == 'dgcnn':
        model = DGCNN_partseg_modify(args).to(device)
    elif args.method == 'gcn':
        print('gcn')
        model = GCN_partseg_modify(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("GPU number is ", torch.cuda.device_count())
    
    abs_file = os.path.dirname(os.path.abspath(__file__)) 
    print(os.path.join(abs_file, args.model_path))
    model.load_state_dict(torch.load(os.path.join(abs_file, args.model_path)))
    model.eval()
    test_acc = 0.0
    count = 0.0
    test_true_seg = []
    test_pred_seg = []
    test_label = []
    cd = 0
    lp = 0
    with torch.no_grad():
        for data, label, seg, gt_pc, _, _ in test_loader:
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg, gt_pc = data.to(device), label_one_hot.to(device), seg.to(device), gt_pc.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred, denoise_pc = model(data, label_one_hot)
            count += 1
            denoise_pc += data
            denoise_pc = denoise_pc.permute(0, 2, 1).contiguous()
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            #denoise_pc = torch.bmm(denoise_pc.permute(0, 2, 1), trans.permute(0, 2, 1))
            #pc = torch.bmm(pc.permute(0, 2, 1), trans.permute(0, 2, 1))
            _, cd_loss, lp_loss = syn_loss(denoise_pc, gt_pc, seg_pred.view(-1, args.seg_number), seg.view(-1,1).squeeze(), args.k)
            cd += cd_loss
            lp += lp_loss
            pred = seg_pred.max(dim=2)[1]
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label.append(label.reshape(-1))

    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label = np.concatenate(test_label)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label, args.class_choice)
    outstr = 'Mean cd loss: %.6f, Mean lp loss: %.6f, Test iou: %.6f' % (cd / count, lp/ count, np.mean(test_ious))
    io.cprint(outstr)
    return np.mean(test_ious)

def train(args, io):
    if args.model_name == 'random0.3':
        train_loader = DataLoader(random_noise_shapenet(num_points=args.num_points, partition='train', noise_min = 0.05, noise_max = args.noise_level), 
                                num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(random_noise_shapenet(num_points=args.num_points, partition='val', noise_min = 0.05, noise_max = args.noise_level), 
                                num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        train_loader = DataLoader(double_ShapeNet(num_points=args.num_points, partition='train', level = args.noise_level), 
                                num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(double_ShapeNet(num_points=args.num_points, partition='val', level = args.noise_level), 
                                num_workers=1, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.draw_loss:
        writer = SummaryWriter(log_dir='losscurve/%s' % args.model_name)
    #Try to load models
    if args.method == 'dgcnn':
        model = DGCNN_partseg_modify(args).to(device)
    elif args.method == 'gcn':
        print('gcn')
        model = GCN_partseg_modify(args).to(device)
    else:
        raise Exception("Not implemented")
    #print(str(model))
    model = nn.DataParallel(model)
    print("GPU number is ", torch.cuda.device_count())
    if args.use_sgd:
        print("Use SGD")
        if args.scheduler == 'cos':
            opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        else:
            opt = optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr*100}], lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.001)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    best_test_iou = 0
    #best_test_iou = test(args, io)
    #print(best_test_iou)
    #if best_test_iou > 0.5:
    #   abs_file = os.path.dirname(os.path.abspath(__file__)) 
     #   model.load_state_dict(torch.load(os.path.join(abs_file, args.model_path)))
    save_file = 'train_show'
    if os.path.exists(save_file) == 0:
        os.mkdir(save_file)
    save_file = os.path.join(save_file, str(args.noise_level))
    
    if os.path.exists(save_file) == 0:
        os.mkdir(save_file)
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_seg = []
        train_pred_seg = []
        train_label = []
        criterion = syn_loss
        chamfer_loss = 0.0
        lp_loss = 0.0
        cnt = 0
        print('lambda: {:}'.format(args.Lambda))
        for data, label, seg, gt_pc, gt_label, gt_seg in train_loader:
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))

            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            gt_pc, gt_seg = gt_pc.to(device), gt_seg.to(device)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred, pc = model(data, label_one_hot)

            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pc = pc + data #
            pc = pc.permute(0, 2, 1).contiguous()
            #pc = torch.bmm(pc.permute(0, 2, 1), trans.permute(0, 2, 1))
            loss_1, loss_2, loss_3 = criterion(pc, gt_pc, seg_pred.view(-1, args.seg_number), seg.view(-1,1).squeeze(),args.k)
            chamfer_loss += loss_2
            lp_loss += loss_3
            cnt = cnt + 1
            assert torch.sum(label == gt_label) == data.shape[0]
            assert torch.sum(seg == gt_seg) == data.shape[0] * args.num_points
            '''print(pc[0])
            print(gt_pc[0])
            a1, a2 = chamfer_distance(torch.unsqueeze(pc[1], dim = 0), torch.unsqueeze(gt_pc[1], dim = 0))
            print(a1)'''
            if cnt % 100 == 1:
                print("epoch: {:}, cnt: {:}, ce loss : {:}, cd loss : {:}, lp loss : {:}".format(epoch, cnt, loss_1, loss_2, loss_3))
            #exit(0)
            loss = loss_1 + loss_2 * 2.0 + loss_3 * 0.05
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item()
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label.append(label.reshape(-1))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label = np.concatenate(train_label)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label, args.class_choice)
        mean_chamfer_loss = chamfer_loss / cnt
        train_loss /= cnt
        outstr = 'Train %d, loss_1: %.6f, loss_2: %.6f, loss_3: %.6f, train iou: %.6f' % (
                    epoch, train_loss, mean_chamfer_loss, lp_loss / cnt, np.mean(train_ious))

        if epoch % 5 == 0:
            io.cprint(outstr)
        else:
            print(outstr)     
        if args.draw_loss:
            writer.add_scalar('chamfer_loss_mean', mean_chamfer_loss, epoch)
            writer.add_scalar('chamfer_loss_total', chamfer_loss, epoch)
        ####################
        # Test
        ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_true_seg = []
        val_pred_seg = []
        val_label = []
        criterion = cal_loss
        with torch.no_grad():
            for data, label, seg, _, _, _ in val_loader:
                label_one_hot = np.zeros((label.shape[0], 16))
                for idx in range(label.shape[0]):
                    label_one_hot[idx, label[idx]] = 1
                label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
                data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred, _ = model(data, label_one_hot)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, args.seg_number), seg.view(-1,1).squeeze())
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                val_true_seg.append(seg_np)
                val_pred_seg.append(pred_np)
                val_label.append(label.reshape(-1))
        val_true_seg = np.concatenate(val_true_seg, axis=0)
        val_pred_seg = np.concatenate(val_pred_seg, axis=0)
        val_label = np.concatenate(val_label)
        test_ious = calculate_shape_IoU(val_pred_seg, val_true_seg, val_label, args.class_choice)
        outstr = 'Val %d, loss: %.6f, val iou: %.6f' % (epoch, val_loss*1.0/count, np.mean(test_ious))
        if epoch % 5 == 0:
            io.cprint(outstr)
        else:
            print(outstr)  
        if np.mean(test_ious) >= best_test_iou and args.save_model == True:
            best_test_iou = np.mean(test_ious)
            print('save {:}'.format(epoch))
            torch.save(model.state_dict(), args.model_path)
    if args.draw_loss:
        writer.close()
    outstr = 'val best iou: %.6f' % (best_test_iou)
    io.cprint(outstr)
   
        

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--decay', type = float, default = 1.0, help = 'weight decay')
    parser.add_argument('--Lambda', type = float, default = 1.0, help = 'loss1 and loss2 partition?')
    parser.add_argument('--save_model', action = 'store_false', default = True, help = 'draw loss quxian?')
    parser.add_argument('--draw_loss', action = 'store_true', default = False, help = 'draw loss quxian?')
    parser.add_argument('--noise_level', type = float, default = 0, help = 'noise level?')
    parser.add_argument('--train', action = 'store_true', default = False, help = 'does train the model?')
    parser.add_argument('--eval', action = 'store_false', default = True, help = 'does test the model?')
    parser.add_argument('--method', type = str, default = 'dgcnn', help = 'method')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--seg_number', type=int, default=50, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_name', type=str, default='dgcnn', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--model_path', type = str, default='my_checkpoints_gcn/random0.3/model.t7', help = 'model relative path for test')
    parser.add_argument('--log_path', type = str, default='my_checkpoints_gcn/', help = 'log relative path for test')
    parser.add_argument('--show_denoise', action = 'store_true', default = False, help = 'show denoised point cloud')
    args = parser.parse_args()


    io = IOStream(args.log_path + args.model_name)
    io.cprint(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if args.train:
        train(args, io)
    if args.eval:
        test(args, io)
    if args.show_denoise:
        show_denoise(args, io)