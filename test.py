from LightField import generate_viewpoints
from net.NetworkIRVAE import IRVAE
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from opt import config_parser
from utils import *
import os
import glob
import argparse
import torch
import numpy as np

def test_scene(model, LF, query_viewpoints, input_idx, cropping=0):
    images = LF[input_idx]
    A, C, H, W = images.size()
    images = images.view(1, -1, H, W).to(query_viewpoints.device)

    pad_input, pad_output = padding(H, W)
    images = torch.nn.functional.pad(images, pad_input, 'replicate')

    rep = model.inference_net.encoder1(images)

    psnr = []
    ssim = []
    for i, c in enumerate(query_viewpoints):
        if i in input_idx:
            continue

        z_inf = model.inference_net.encoder2(rep, c)
        output = model.inference_net.decoder(images, z_inf)

        output = torch.nn.functional.pad(output, pad_output)
        output = torch.clamp(output, 0, 1)

        gt = LF[i].permute(1, 2, 0).numpy()[cropping:-cropping, cropping:-cropping]
        pred = output[0].permute(1, 2, 0).cpu().numpy()[cropping:-cropping, cropping:-cropping]

        psnr_img = compare_psnr(gt, pred, data_range=1)
        # ssim_img = compare_ssim(gt, pred, data_range=1) # produce slightly different result
        ssim_img = compare_ssim(gt, pred, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True) # in line with some compared methods

        psnr.append(psnr_img)
        ssim.append(ssim_img)

    return np.mean(psnr), np.mean(ssim)

if __name__ == '__main__':
    args = config_parser()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    model = IRVAE(args)
    model.load_state_dict(torch.load(args.state_path, map_location='cpu'))
    model = model.eval()
    model = model.to(device)

    with torch.no_grad():
        filelist = glob.glob(args.testing_data_path)
        query_viewpoints = generate_viewpoints(args.num_out_views).to(device)
        input_idx = args.input_idx

        if filelist[0][-3:] == 'mat':
            read_lf = read_mat
        else:
            read_lf = read_eslf

        res_psnr = np.zeros((len(filelist)))
        res_ssim = np.zeros((len(filelist)))
        res_time = np.zeros((len(filelist)))
        for k, filepath in enumerate(filelist):
            filename = os.path.basename(filepath)[:-4]
            LF = read_lf(filepath, args.lf_start_idx, args.lf_end_idx)

            res_psnr[k], res_ssim[k] = test_scene(model, LF, query_viewpoints, input_idx, cropping=22)
            print('{}, {}, PSNR:{}\tSSIM:{}'.format(k, filename, res_psnr[k], res_ssim[k]))

        print('Average PSNR:{}\tSSIM:{}'.format(res_psnr.mean(), res_ssim.mean()))
