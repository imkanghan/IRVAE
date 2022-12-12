# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from  adpconv.adaptiveconv import AdaptiveConv

import net.hrnet as hrnet
from net.model_config import MODEL_EXTRAS

FILTER_LENGTH = 15

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return torch.cat([x, self.lrelu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1, padding=0)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class RDNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_rdbs=12):
        super(RDNEncoder, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=7, padding=3)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        rdbs = []
        for i in range(num_rdbs):
            rdbs.append(RDB(64, 64, 6))

        self.rdbs = torch.nn.ModuleList(rdbs)

        self.gff = nn.Conv2d(64 * (num_rdbs+1), 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x  = F.leaky_relu(self.conv0(x), negative_slope = 0.1)
        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        x  = F.leaky_relu(self.conv2(x), negative_slope = 0.1)

        features = [x]
        for i in range(len(self.rdbs)):
            x = self.rdbs[i](x)
            features.append(x)

        out = self.gff(torch.cat(features, dim=1))
        out = self.conv3(out)
        return out

# Map the input and output channels by two conv layers
class RDBMap(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDBMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.rdb = RDB(64, 64, 6)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        x = self.rdb(x)
        x = self.conv2(x)

        return x

class RepEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.rdb = RDB(64, out_channels, 6)

    def forward(self, r, c):
        batch_size, _, *dims = r.size()
        c_expanded = c.view(batch_size, -1, 1, 1).repeat(1, 1, *dims)
        rc = torch.cat([r, c_expanded], dim=1)

        rc = F.leaky_relu(self.conv1(rc), negative_slope = 0.1)
        out = self.rdb(rc)

        return out


class Decoder(torch.nn.Module):
    def __init__(self, color_channels):
        super(Decoder, self).__init__()

        self.kernel_decoder = RDBMap(64, 4 * FILTER_LENGTH * FILTER_LENGTH)
        self.bias_decoder = RDBMap(64, color_channels)

        self.adpconv = AdaptiveConv()

        self.pad = torch.nn.ReplicationPad2d([int(math.floor(FILTER_LENGTH
                / 2.0)), int(math.floor(FILTER_LENGTH / 2.0)), int(math.floor(FILTER_LENGTH
                / 2.0)), int(math.floor(FILTER_LENGTH / 2.0))])

    def forward(self, xr, z):
        bias = self.bias_decoder(z)

        kernels = self.kernel_decoder(z)
        kernels = torch.softmax(kernels, dim=1)
        k1, k2, k3, k4 = torch.chunk(kernels, chunks=4, dim=1)

        img1, img2, img3, img4 = torch.chunk(self.pad(xr), chunks=4, dim=1)

        warped1 = self.adpconv(img1, k1)
        warped2 = self.adpconv(img2, k2)
        warped3 = self.adpconv(img3, k3)
        warped4 = self.adpconv(img4, k4)

        out = warped1 + warped2 + warped3 + warped4 + bias
        # out = bias
        return out

class InferenceNet(torch.nn.Module):
    def __init__(self, args):
        super(InferenceNet, self).__init__()

        if args.encoder == 'hrnet':
            cfg = MODEL_EXTRAS['seg_hrnet']
            cfg['in_channels'] = args.inf_in_channels
            cfg['HIGH_RESOLUTION_NET.DATASET.NUM_CLASSES'] = args.rep_channels
            self.encoder1 = hrnet.get_seg_model(cfg)
        elif args.encoder == 'rdn':
            self.encoder1 = RDNEncoder(args.inf_in_channels, args.rep_channels)
        else:
            raise Exception("Only hrnet and rdn are supported for encoder")


        self.encoder2 = RepEncoder(args.rep_channels+2, args.z_channels)
        self.decoder = Decoder(args.color_channels)

    def forward(self, xr, c):
        r = self.encoder1(xr)
        z_inf = self.encoder2(r, c)

        xn_inf = self.decoder(xr, z_inf)

        return xn_inf, z_inf

class ReconstructionNet(torch.nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()

        if args.encoder == 'hrnet':
            self.encoder = RDBMap(args.rec_in_channels, args.z_channels)
        elif args.encoder == 'rdn':
            self.encoder = RDNEncoder(args.rec_in_channels, args.rep_channels, num_rdbs=1)
        else:
            raise Exception("Only hrnet and rdn are supported for encoder")

        self.decoder = Decoder(args.color_channels)

    def forward(self, xr, xn):
        z_rec = self.encoder(torch.cat((xr, xn), dim=1))
        xn_rec = self.decoder(xr, z_rec)

        return xn_rec, z_rec

# Inference-Reconstruction Variational Autoencoder (IRVAE)
class IRVAE(torch.nn.Module):
    def __init__(self, args):
        super(IRVAE, self).__init__()

        self.inference_net = InferenceNet(args)
        self.reconstruction_net = ReconstructionNet(args)

    def forward(self, xr, xn, c):

        xn_inf, z_inf = self.inference_net(xr, c)
        xn_rec, z_rec = self.reconstruction_net(xr, xn)

        return xn_inf, xn_rec, z_inf, z_rec
