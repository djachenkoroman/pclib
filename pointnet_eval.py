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

parser = argparse.ArgumentParser()
parser.add_argument('--dsfile', type=str, default='', help='dataset file [default=null]')
parser.add_argument('--modfile', type=str, default='', help='models directory [default=null]')
parser.add_argument('--gridsize', type=int, default=50, help='gridsize [default=50]')
parser.add_argument('--npoints', type=int, default=2400, help='npoints [default=2400]')
parser.add_argument('--num_classes', type=int, default=5, help='num_classes [default=5]')
parser.add_argument('--channels', type=int, default=3, help='channels [default=3]')
parser.add_argument('--device', type=str, default='cpu', help='current device [default=cuda]')

args = parser.parse_args()

if __name__ == '__main__':
    tprint("pointnet evaluate")
