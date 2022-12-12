import collections, os, io
import torch
import glob
import numpy as np
import random
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from utils import *

num_img_x = 14
num_img_y = 14
height = 5264 // 14
width = 7574 // 14
patch_size = 64

def process_training_batch(images, viewpoints, representation_idx=[0, 7, 56, 63]):
    """
    Randomly select a view in LF as the training target
    """
    batch_size, m, *_ = viewpoints.size()

    indices = torch.randperm(m)
    query_idx = random.randint(0, m-1)

    xr, c_xr = images[:, representation_idx], viewpoints[:, representation_idx]
    # Use random (image, viewpoint) pair in batch as target
    xn, c_xn = images[:, query_idx], viewpoints[:, query_idx]

    return xr, c_xr, xn, c_xn

def generate_viewpoints(split_num=8):
    """
    Generate mesh viewpoints
    :param split_num: how many viewpoints in the row and column
    """
    cameras = torch.zeros(split_num, split_num, 2)
    grid_y, grid_x = torch.meshgrid((torch.linspace(0.0, 1.0, split_num), torch.linspace(0.0, 1.0, split_num)))
    cameras[:, :, 0] = grid_y
    cameras[:, :, 1] = grid_x
    cameras = cameras.view(-1, 2)
    return cameras


class LightField(Dataset):
    def __init__(self, root_dir, lf_start_idx, lf_end_idx, res_out=8, color_channels=3, step_size=16):
        self.root_dir = root_dir
        self.step_size = step_size

        self.viewpoints = generate_viewpoints(res_out)

        filelist = []
        if isinstance(self.root_dir, list):
            for d in self.root_dir:
                filelist += glob.glob(d)
        else:
            filelist = glob.glob(self.root_dir)

        self.LF = torch.zeros(len(filelist), res_out * res_out, color_channels, height, width)
        self.length = len(filelist) * ((height - patch_size) // self.step_size + 1) * ((width - patch_size) // self.step_size + 1)

        if filelist[0][-3:] == 'mat':
            read_lf = read_mat
        else:
            read_lf = read_eslf

        for i, filepath in enumerate(filelist):
            print("Reading light field image {} : {}".format(i, os.path.basename(filepath)))
            LF = read_lf(filepath, lf_start_idx, lf_end_idx)
            _, _, h, w = LF.size()
            self.LF[i, :, :, 0:h, 0:w] = LF

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_idx, y, x = self.__getindex__(idx)

        images = self.LF[image_idx, :, :, y*self.step_size : y*self.step_size + patch_size, x*self.step_size : x*self.step_size + patch_size]
        viewpoints = self.viewpoints

        return images, viewpoints

    def __getindex__(self, idx):
        num_row = (height - patch_size) // self.step_size + 1
        num_column = (width - patch_size) // self.step_size + 1

        image_idx = idx // (num_row * num_column)
        y = idx % (num_row * num_column) // num_column
        x = (idx % (num_row * num_column)) % num_column

        return image_idx, y, x
