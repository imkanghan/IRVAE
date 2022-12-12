import torch
import scipy.io as sio
import numpy as np
from PIL import Image

num_img_x = 14
num_img_y = 14

def read_eslf(filepath, lf_start_idx, lf_end_idx):
    img = Image.open(filepath)
    img = img.convert("RGB")
    w = img.size[0] // num_img_x
    h = img.size[1] // num_img_y
    img = np.asarray(img) # h, w, c

    LF = torch.zeros(h, w, 3, num_img_y, num_img_x)
    for ax in range(num_img_x):
        for ay in range(num_img_y):
            LF[:, :, :, ay, ax] = torch.tensor(img[ay::num_img_y, ax::num_img_x])
    LF = LF[:, :, :, lf_start_idx:lf_end_idx, lf_start_idx:lf_end_idx] # h, w, c, u, v
    LF = LF.permute(3, 4, 2, 0, 1) #u, v, c, h, w
    _, _, *CHW= LF.size()
    LF = LF.contiguous().view(-1, *CHW)
    return LF / 255.0

def read_mat(filepath, lf_start_idx, lf_end_idx):
    img = sio.loadmat(filepath)['img_y']
    h = img.shape[0] // num_img_y
    w = img.shape[1] // num_img_x

    LF = torch.zeros(h, w, 1, num_img_y, num_img_x)
    for ax in range(num_img_x):
        for ay in range(num_img_y):
            LF[:, :, :, ay, ax] = torch.tensor(img[ay::num_img_y, ax::num_img_x])[..., None]
    LF = LF[:, :, :, lf_start_idx:lf_end_idx, lf_start_idx:lf_end_idx] # h, w, c, u, v for 2x2 -> 7x7
    LF = LF.permute(3, 4, 2, 0, 1) #u, v, c, h, w
    _, _, *CHW= LF.size()
    LF = LF.contiguous().view(-1, *CHW)

    return LF

def padding(H, W, t=5):
    pad_h = 0
    if (H - (H >> t << t)) != 0:
        pad_h = (1<<t) - (H - (H >> t << t))

    pad_w = 0
    if (W - (W >> t << t)) != 0:
        pad_w = (1<<t) - (W - (W >> t << t))

    pad_input = (0, pad_w, 0, pad_h)
    pad_output = (0, -pad_w, 0, -pad_h)

    return pad_input, pad_output
