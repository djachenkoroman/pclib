import math
import os
import sys
import gdown
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.optim as optim
import pandas as pd
import torch
from pointnet import PointNetDenseCls
from pcvis import pcshow_xyz,pcshow_xyzl,pcshow_xyzrgb
from utils import preprocess,pointcloud_pointnet_seg_real,pointcloud_pointnet_seg
from terra import Terra,TerraRGB
import argparse
from art import *

parser = argparse.ArgumentParser()
parser.add_argument('--dsfile', type=str, default='', help='dataset file [default=null]')
parser.add_argument('--moddir', type=str, default='', help='models directory [default=null]')
parser.add_argument('--dsdir', type=str, default='', help='dataset directory [default=null]')
parser.add_argument('--gridsize', type=int, default=50, help='gridsize [default=50]')
parser.add_argument('--npoints', type=int, default=2400, help='npoints [default=2400]')
parser.add_argument('--channels', type=int, default=3, help='channels [default=3]')
parser.add_argument('--device', type=str, default='cuda', help='current device [default=cuda]')
parser.add_argument('--epochs', type=int, default=10, help='epochs [default=10]')

args = parser.parse_args()

params = {
    'batch_size':4,
    'num_workers':1,
    'shuffle':True,
    'epochs':args.epochs,
    'lr':0.001,
    'npoints':args.npoints,
}

# fn_templ='/content/models/model_pointnet_ch{0}_np{1}_gs{2}_ep{3}_{4}_acc{5}'
fn_templ='/content/models/model_pointnet_ch{0}_gs{1}_nc{2}_np{3}_ep{4}({5})_acc{6}'

if __name__ == '__main__':
    tprint("pointnet train")
    print("dataset file: {0}".format(args.dsfile))
    if not os.path.isfile(args.dsfile):
        sys.exit("dsfile not found")

    print("models directory: {0}".format(args.moddir))
    print("dataset directory: {0}".format(args.dsdir))

    if not os.path.isdir(args.moddir):
        sys.exit("moddir not found")
    # if not os.path.isdir(args.dsdir):
    #     sys.exit("dsdir not found")

    print("gridsize: {0}".format(args.gridsize))
    print("channels: {0}".format(args.channels))
    print("device: {0}".format(args.device))
    print("epochs: {0}".format(args.epochs))
    print("npoints: {0}".format(params['npoints']))

    num_classes, classes = preprocess(os.path.join(args.dsfile), os.path.join(args.dsdir), args.gridsize)
    if args.channels==3:
        train_dataset = Terra(args.dsdir, data_augmentation=True)
        test_dataset = Terra(args.dsdir, split='test')
    elif args.channels==6:
        train_dataset = TerraRGB(args.dsdir, data_augmentation=True)
        test_dataset = TerraRGB(args.dsdir, split='test')
    else:
        sys.exit("incorrect channel value")


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                                   shuffle=params['shuffle'], num_workers=params['num_workers'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'],
                                                  shuffle=params['shuffle'], num_workers=params['num_workers'])
    classifier = PointNetDenseCls(channels=args.channels, num_classes=num_classes)
    optimizer = optim.Adam(classifier.parameters(), lr=params['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.to(args.device)

    num_batch = len(train_dataset) / params['batch_size']

    for epoch in range(params['epochs']):
        acc = 0
        for i, data in enumerate(train_dataloader):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(params['batch_size'] * params['npoints'])))

            if i % 10 == 0:
                j, data = next(enumerate(test_dataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.to(args.device), target.to(args.device)
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                acc = correct.item() / float(params['batch_size'] * params['npoints'])
                print('[%d: %d/%d] loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), acc))
        scheduler.step()
        # fn_templ = '/content/models/model_pointnet_ch{0}_np{1}_gs{2}_ep{3}_{4}_acc{5}'
        # fn_templ = '/content/models/model_pointnet_ch{0}_gs{1}_nc{2}_np{3}_ep{4}_{5}_acc{6}'

        torch.save(classifier.state_dict(), fn_templ.format(args.channels,args.gridsize,num_classes,args.npoints,str(epoch).zfill(4),args.epochs, round(acc, 4)))
