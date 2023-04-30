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


if __name__ == __main__:
    pass