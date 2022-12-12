import sys
import torch
import torchvision
from LightField import LightField, process_training_batch
from torch.utils.data import DataLoader
from net.NetworkIRVAE import IRVAE
from opt import config_parser

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def compute_kernel(x, y, local_size):
    N, C, H, W = x.size()
    x = x.view(N, -1, H//local_size, W//local_size)
    y = y.view(N, -1, H//local_size, W//local_size)

    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim, -1, -1)
    tiled_y = y.expand(x_size, y_size, dim, -1, -1)

    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mlmmd(x, y, local_size):
    x_kernel = compute_kernel(x, x, local_size)
    y_kernel = compute_kernel(y, y, local_size)
    xy_kernel = compute_kernel(x, y, local_size)
    mlmmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mlmmd

class VggLoss(torch.nn.Module):
    def __init__(self, device):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        # if only y channel
        if output.size(1) == 1:
            output = output.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)

        outputFeatures = self.features(output)
        targetFeatures = self.features(target)
        loss = torch.mean((outputFeatures - targetFeatures).pow(2))

        return loss


if __name__ == '__main__':
    args = config_parser()

    #define network
    network = IRVAE(args)
    network = network.to(device)
    network = torch.nn.DataParallel(network)

    #define light field dataloader
    dataset = LightField(args.training_data_path, args.lf_start_idx, args.lf_end_idx, args.num_out_views, args.color_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    epochs = args.epochs
    lr = 1e-4
    lr_target_ratio = 0.001
    lr_factor = lr_target_ratio**(1 / epochs)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    print_iter = 100

    alpha = 1.0
    eta = 0.1
    beta = 0.1
    gamma = 10.0
    print('Loss weights: alpha:', alpha, 'eta:', eta, 'beta:', beta, 'gamma:', gamma)

    loss = torch.nn.L1Loss()
    vgg = VggLoss(device).to(device)
    total_iter = 0
    for epoch in range(epochs):
        sum_loss = 0
        sum_l1 = 0
        sum_lvgg = 0
        sum_mlmmd = 0

        sum_l1_inf = 0
        sum_l1_rec = 0
        sum_lvgg_inf = 0
        sum_lvgg_rec = 0
        for idx, (images, viewpoints) in enumerate(loader):
            images = images.to(device)
            viewpoints = viewpoints.to(device)

            xr, c_xr, xn, c_xn = process_training_batch(images, viewpoints, representation_idx=args.input_idx)
            N, A, C, H, W = xr.size()
            xr = xr.view(N, -1, H, W)

            xn_inf, xn_rec, z_inf, z_rec = network(xr, xn, c_xn)

            l1_inf = loss(xn_inf, xn)
            l1_rec = loss(xn_rec, xn)
            lvgg_inf = vgg(xn_inf, xn)
            lvgg_rec = vgg(xn_rec, xn)
            mlmmd = compute_mlmmd(z_inf, z_rec, 8)

            l1 = l1_inf + eta * l1_rec
            lvgg = lvgg_inf + eta / beta * lvgg_rec
            L = alpha * l1 + beta * lvgg + gamma * mlmmd

            sum_l1_inf += l1_inf.item()
            sum_l1_rec += l1_rec.item()
            sum_lvgg_inf += lvgg_inf.item()
            sum_lvgg_rec += lvgg_rec.item()

            sum_loss += L.item()
            sum_l1 += l1.item()
            sum_lvgg += lvgg.item()
            sum_mlmmd += mlmmd.item()

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx %  print_iter == 0 and idx != 0:
                print('Train Epoch: {} [{}/{}]\t lr:{:.4f}\t L:{:06f}\t L1: {:.6f}\tVGG:{:.6f}\t MLMMD: {:.6f}'.format(
                    epoch, idx, len(dataset) // args.batch_size, lr,
                    sum_loss / print_iter,
                    sum_l1 / print_iter,
                    sum_lvgg / print_iter,
                    sum_mlmmd / print_iter))

                print('L1_inf: {:.6f}\tL1_rec:{:.6f}\t VGG_inf:{:.6f}\tVGG_rec:{:.6f}'.format(
                    sum_l1_inf / print_iter,
                    sum_l1_rec / print_iter,
                    sum_lvgg_inf / print_iter,
                    sum_lvgg_rec / print_iter))

                sum_loss = 0
                sum_l1 = 0
                sum_lvgg = 0
                sum_mlmmd = 0

                sum_l1_inf = 0
                sum_l1_rec = 0
                sum_lvgg_inf = 0
                sum_lvgg_rec = 0

        lr = lr * lr_factor
        optimizer.lr = lr


        if epoch > 100:
            torch.save(network.module.state_dict(),
                    "model_states/lytro-2x2-{}x{}-{}-{}-encoder-{}.pt".format(args.num_out_views, args.num_out_views, args.color, args.encoder, epoch))
