import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import os
import logging
import numpy as np
from torch.utils.data import Dataset


class Terra(Dataset):
    """Terra dataset"""

    def __init__(self,
                 data_dir: str = 'terra_dataset',
                 test_size: float = 0.2,
                 split: str = 'train',
                 loop: int = 1,
                 grid_size: int = 10,
                 shuffle: bool = True,
                 npoints=2400,
                 data_augmentation=True,
                 ):

        super().__init__()
        self.npoints = npoints
        self.split = split
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.data_augmentation = data_augmentation

        data_list = sorted(os.listdir(data_dir))

        if split == 'train':
            self.data_list = data_list[:int(len(data_list) * (1 - test_size))]
        else:
            self.data_list = data_list[-int(len(data_list) * test_size):]

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        data_path = os.path.join(
            self.data_dir, self.data_list[data_idx])
        cdata = np.load(data_path).astype(np.float32)
        choice = np.random.choice(len(cdata), self.npoints, replace=True)
        cdata = cdata[choice, :]
        coord, label = cdata[:, :3], cdata[:, -1:]

        coord = coord - np.expand_dims(np.mean(coord, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(coord ** 2, axis=1)), 0)
        coord = coord / dist  # scale

        if self.data_augmentation:
            coord += np.random.normal(0, 0.02, size=coord.shape)  # random jitter

        label = label.squeeze(-1).astype(np.compat.long)
        label -= 2
        data = np.hstack([coord, rgb]).astype(np.float32)
        return torch.tensor(data), torch.tensor(label)

    def __len__(self):
        return len(self.data_idx)
