import os
from path import Path
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
sys.path.append('/content/pclib')
from pcvis import pcshow_xyz,pcshow_xyzl,pcshow_xyzrgb
from utils import preprocess,preprocess2,preprocess3,pointcloud_pointnet_seg
from terra import Terra,TerraRGB
from pointnet import PointNetDenseCls
import argparse
from art import *
import random
import string

parser = argparse.ArgumentParser()
parser.add_argument('--dsfile', type=str, default='', help='dataset file [default=null]')
parser.add_argument('--modfile', type=str, default='', help='models directory [default=null]')
parser.add_argument('--gridsize', type=int, default=50, help='gridsize [default=50]')
parser.add_argument('--npoints', type=int, default=2400, help='npoints [default=2400]')
parser.add_argument('--num_classes', type=int, default=5, help='num_classes [default=5]')
parser.add_argument('--channels', type=int, default=3, help='channels [default=3]')
parser.add_argument('--device', type=str, default='cpu', help='current device [default=cuda]')

args = parser.parse_args()

msg_tmpl='''
dsfile: {0} (exists !!!)
modfile: {1} (exists !!!)
gridsize: {2}
npoints: {3}
num_classes: {4}
channels: {5}
device: {6}
'''

if __name__ == '__main__':
    tprint("pointnet evaluate")

    if not os.path.isfile(args.dsfile):
        sys.exit("dsfile not found")

    if not os.path.isfile(args.modfile):
        sys.exit("modfile not found")

    print(msg_tmpl.format(args.dsfile,args.modfile,args.gridsize,args.npoints,args.num_classes,args.channels,args.device))

    current_dir=os.getcwd()
    print(f'current dir: {current_dir}')

    rand_string = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    data_dir=os.path.join(current_dir,rand_string)
    os.makedirs(data_dir, exist_ok=False)
    print(f'data dir: {data_dir}')

    data = np.loadtxt(args.dsfile)
    print(f'file {args.dsfile} downloaded')
    data = data[:,:args.channels]
    if args.channels==3:
        fmt = '%1.6f', '%1.6f', '%1.6f'
    elif args.channels==6:
        fmt = '%1.6f', '%1.6f', '%1.6f', '%d', '%d', '%d'
    else:
        sys.exit("error")

    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y