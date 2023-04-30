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
import os
import pickle
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.optim as optim
import os
import pandas as pd
import torch
from pointnet import PointNetDenseCls
from pcvis import pcshow_xyz,pcshow_xyzl,pcshow_xyzrgb
from utils import preprocess,pointcloud_pointnet_seg_real,pointcloud_pointnet_seg
from terra import Terra,TerraRGB
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='current device [default=cuda]')
parser.add_argument('--moddir', type=str, default='', help='models directory [default=null]')
parser.add_argument('--epochs', type=int, default=10, help='epochs [default=10]')

args = parser.parse_args()

params = {
    'batch_size':4,
    'num_workers':1,
    'shuffle':True,
    'output_file':'terra_model',
    'epochs':args.epochs,
    'lr':0.001
}


if __name__ == '__main__':
    print("device: {0}".format(args.device))
    print("models directory: {0}".format(args.moddir))

    if not os.path.isdir(args.moddir):
        sys.exit("moddir not found")

    print("epochs: {0}".format(args.moddir))
