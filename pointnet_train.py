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
import datetime
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--dsfile', type=str, default='', help='dataset file [default=null]')
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

fn_templ='model_{1}_pointnet_ch{2}_gs{3}_nc{4}_np{5}_ep{6}({7})_acc{8}'
dt_templ="{0}{1}{2}{3}{4}"
s_templ = "[{0}: {1}/{2}] train loss: {3} accuracy: {4}"
input_params_templ = '''
Current DIR: {0}
DATE ID: {1}
dataset file: {2}
models directory: {3}
dataset directory: {4}
gridsize: {5}
channels: {6}
device: {7}
epochs: {8}
npoints: {9}
num_classes: {10}
'''

if __name__ == '__main__':
    tprint("pointnet train")

    # Get current DIR
    maindir=os.getcwd()

    # Get DATE and TIME
    date_time = datetime.datetime.now()
    date_id=dt_templ.format(date_time.year, str(date_time.month).zfill(2), str(date_time.day).zfill(2), str(date_time.hour).zfill(2), str(date_time.minute).zfill(2))


    moddir=os.path.join(maindir,"models_{0}".format(date_id))
    os.makedirs(moddir, exist_ok=False)

    dsdir=os.path.join(maindir,"dsdir_{0}".format(date_id))
    os.makedirs(dsdir, exist_ok=False)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(moddir,"log_{0}.log").format(date_id), filemode="w")

    if not os.path.isfile(args.dsfile):
        logging.info("dsfile not found")
        sys.exit("dsfile not found")

    num_classes, classes = preprocess(os.path.join(args.dsfile), os.path.join(dsdir), args.gridsize)
    if args.channels==3:
        train_dataset = Terra(dsdir, data_augmentation=True)
        test_dataset = Terra(dsdir, split='test')
    elif args.channels==6:
        train_dataset = TerraRGB(dsdir, data_augmentation=True)
        test_dataset = TerraRGB(dsdir, split='test')
    else:
        logging.info("incorrect channel value")
        sys.exit("incorrect channel value")

    str_pr=input_params_templ.format(
        maindir,
        date_id,
        args.dsfile,
        moddir,
        dsdir,
        args.gridsize,
        args.channels,
        args.device,
        args.epochs,
        args.npoints,
        num_classes
    )

    print(str_pr)
    logging.info(str_pr)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                                   shuffle=params['shuffle'], num_workers=params['num_workers'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'],
                                                  shuffle=params['shuffle'], num_workers=params['num_workers'])
    classifier = PointNetDenseCls(channels=args.channels, num_classes=num_classes)
    optimizer = optim.Adam(classifier.parameters(), lr=params['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.to(args.device)

    num_batch = len(train_dataset) / params['batch_size']
    m_loss=[]
    m_accuracy=[]
    for epoch in tqdm(range(params['epochs']),ncols=100):
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
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            # epoch, i, num_batch, loss.item(), correct.item() / float(params['batch_size'] * params['npoints'])))
            # tqdm.write(s_templ.format(epoch, i, num_batch, loss.item(), correct.item() / float(params['batch_size'] * params['npoints'])))

            logging.info(s_templ.format(epoch, i, num_batch, loss.item(), correct.item() / float(params['batch_size'] * params['npoints'])))

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
                # print('[%d: %d/%d] loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), acc))
                # tqdm.write(s_templ.format(epoch, i, num_batch, loss.item(), loss.item(),acc))
                logging.info(s_templ.format(epoch, i, num_batch, loss.item(), loss.item(),acc))
        scheduler.step()
        m_loss.append(loss.item())
        m_accuracy.append(acc)
        fnm=fn_templ.format(moddir,date_id, args.channels, args.gridsize, num_classes, args.npoints, str(epoch).zfill(4), args.epochs, round(acc, 4))

        fnm_full=os.path.join(moddir,fnm)
        # tqdm.write(fnm_full)
        torch.save(classifier.state_dict(), fnm_full)
        logging.info("model saved: {0}".format(fnm_full))
    ## benchmark mIOU
    shape_ious = []
    predictions = []

    tprint("test")

    for i,data in enumerate(tqdm(test_dataloader,ncols=100), 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(args.device), target.to(args.device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        accuracy=correct.item()/float(params["batch_size"] * params['npoints'])
        # tqdm.write(f'loss: {loss.item()} accuracy: { accuracy }')
        logging.info(f'loss: {loss.item()} accuracy: { accuracy }')
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy()
        predictions.append((points, pred_np, target_np))

    tprint("result")
    print(f'loss: {loss.item()} accuracy: {accuracy}')
    logging.info(f'result loss: {loss.item()} accuracy: {accuracy}')

    # print(m_accuracy)
    # print(m_loss)

    data_fn=os.path.join(moddir,"data_{0}.txt".format(date_id))
    with open(data_fn, 'w') as filehandle:
        for l,c in zip(m_loss,m_accuracy):
            filehandle.write("{0} {1}\n".format(str(l),str(c)))
        logging.info("file {0} saved".format(data_fn))