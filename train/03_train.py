from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_utils.dataLoader import S3DIS
from models.dgcnn_sem_seg import dgcnn_sem_seg
import numpy as np
from torch.utils.data import DataLoader
from data_utils.util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
import logging
from tqdm import tqdm

global room_seg
room_seg = []
global room_pred
room_pred = []
# global visual_warning
# visual_warning = True

def _init_():
    if not os.path.exists('log/'):
        os.makedirs('log/')
    if not os.path.exists('log/'+args.exp_name):
        os.makedirs('log/'+args.exp_name)
    checkpoint_dir = os.path.join('log/', args.exp_name, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def update_cm(cm, true, pred, n=3):
    cm += np.bincount(n * true.reshape(-1).astype(np.int64) + pred.reshape(-1), minlength=n * n).reshape(n, n)

def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    acc = tp.sum() / max(cm.sum(), 1)
    avg_class_acc = (tp / np.maximum(cm.sum(1), 1)).mean()
    ious = tp / np.maximum(cm.sum(1) + cm.sum(0) - tp, 1)
    return acc, avg_class_acc, ious

def train(args, io): 
    print("start loading training data ...")
    TRAIN_DATASET = S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area)
    print("start loading test data ...")
    TEST_DATASET = S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area)

    import data_utils.dataLoader as dl
    dl._H5_CACHE.clear()

    train_loader = DataLoader(TRAIN_DATASET, num_workers=0, batch_size=args.batch_size, shuffle=True, 
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(TEST_DATASET, num_workers=0, batch_size=args.test_batch_size, shuffle=False, 
                             pin_memory=True, drop_last=False)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    device = torch.device("cuda" if args.cuda else "cpu")

    # Class weights (same formula as PointNet++): rare classes get higher weight
    counts = np.bincount(TRAIN_DATASET.seg.reshape(-1).astype(np.int64), minlength=3).astype(np.float64)
    freq = counts / counts.sum()
    class_weights = torch.tensor(np.power(freq.max() / np.maximum(freq, 1e-12), 1 / 3.0),
                                 dtype=torch.float32, device=device)
    log_string('Class counts: %s, weights: %s' % (counts.astype(int).tolist(), class_weights.cpu().numpy().round(3).tolist()))

    #Try to load models
    if args.model == 'dgcnn':
        model = dgcnn_sem_seg(args).to(device)
    else:
        raise Exception("Not implemented")
    # print(str(model))
    
    model = nn.DataParallel(model)
    print("Using", torch.cuda.device_count(), "GPUs = ", torch.cuda.get_device_name(0))

    #Check pre-trained model
    best_path = os.path.join('log', args.exp_name, 'checkpoint', 'best_model.t7')
    if args.resume and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        log_string('Resumed from %s' % best_path)
    else:
        log_string('Starting training from scratch...')

    # print("Optimizer = Adam")
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss
    LEARNING_RATE_CLIP = 1e-5
    global_epoch = 0
    best_test_iou = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        ####################
        # Train
        ####################
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epochs))
        log_string('Learning rate: %f' % opt.param_groups[0]['lr'])
        
        train_loss = 0.0
        count = 0.0
        model.train()
        train_cm = np.zeros((3, 3), dtype=np.int64)

        for i, (data, seg) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1), weight=class_weights) # 3 Number of classes
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            update_cm(train_cm, seg_np, pred_np)

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_acc, avg_per_class_acc, train_ious = metrics_from_cm(train_cm)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        
        if epoch % 1 == 0:
            logger.info('Save model...')
            savepath = 'log/%s/checkpoint/model.t7' % (args.exp_name)
            log_string('Saving at %s' % savepath)
            torch.save(model.state_dict(), savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_cm = np.zeros((3, 3), dtype=np.int64)

        log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
        for i, (data, seg) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
          with torch.no_grad():
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1), weight=class_weights) # 3 Number of classes
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            update_cm(test_cm, seg_np, pred_np)
        test_acc, avg_per_class_acc, test_ious = metrics_from_cm(test_cm)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        log_string('Per-class IoU [ground, veg, building]: %s' % np.round(test_ious, 4).tolist())
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            logger.info('Save model...')
            savepath = 'log/%s/checkpoint/best_model.t7' % (args.exp_name)
            log_string('Saving at %s' % savepath)
            torch.save(model.state_dict(), savepath)
            log_string('Saving model....')
            log_string('Best mIoU: %f' % best_test_iou)
        global_epoch += 1

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N', choices=['dgcnn'], help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N', choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default='1', help='Area number used for testing, e.g. 1')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', action='store_true', help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action='store_true', help='Resume from best_model.t7 of this exp_name')
    # parser.add_argument('--model_root', type=str, default='', metavar='N', help='Pretrained model root')
    # parser.add_argument('--visu', type=str, default='', help='visualize the model')
    # parser.add_argument('--visu_format', type=str, default='ply', help='file format of visualization')
    args = parser.parse_args()

    _init_()
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_dir = 'log/'+args.exp_name+'/'+'log'
    
    def log_string(str):
        logger.info(str)
        print(str)

    io = IOStream('log/' + args.exp_name + '/run.log')
    log_string('PARAMETER ...')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        # io.cprint(
        #     'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        print("Test Function not defined")