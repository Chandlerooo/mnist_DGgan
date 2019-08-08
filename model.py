import argparse
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epoch of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batch")
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--D_path", type=str, default='parameters/Dist.pkl')
parser.add_argument("--G_path", type=str, default='parameters/Gene.pkl')
opt = parser.parse_args()
# print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  # prod-product
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *img_shape)
        return img


class Discirminator(nn.Module):
    def __init__(self):
        super(Discirminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class CNN_G(nn.Module):
    def __init__(self):
        super(CNN_G, self).__init__()
        self.model = nn.Sequential(
            # 100*1*1 -> (64*4)*4*4
            nn.ConvTranspose2d(100, 64 * 4, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=64 * 4),
            nn.ReLU(True),
            # (64*4)*4*4 -> (64*2)*8*8
            nn.ConvTranspose2d(64 * 4, 64 * 2, stride=2, padding=1,
                               kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=64 * 2),
            nn.ReLU(True),
            # (64*2)*8*8 -> 64*16*16
            nn.ConvTranspose2d(64 * 2, 64, stride=2, padding=1,
                               kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # 64*16*16 -> 1*28*28
            nn.ConvTranspose2d(64, 1, stride=2, padding=3,
                               kernel_size=4, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, opt.latent_dim, 1, 1)
        out = self.model(x)
        out = out.view(out.shape[0], 1, 28, 28)
        return out


class CNN_D(nn.Module):
    def __init__(self):
        super(CNN_D, self).__init__()
        self.model = nn.Sequential(
            # 1*28*28 -> 64*16*16
            nn.Conv2d(1, 64, kernel_size=2,
                      stride=2, padding=3, bias=False),
            nn.LeakyReLU(inplace=True),
            # 64*16*16 -> (64*2)*8*8
            nn.Conv2d(64, 64 * 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(inplace=True),
            # (64*2)*8*8 -> (64*4)*4*4
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(inplace=True),
            # (64*4)*4*4 -> 1*1*1
            nn.Conv2d(64 * 4, 1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        in_img = img.view(-1, 1, 28, 28)
        validity = self.model(in_img).view(-1, 1)
        return validity
