import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

import model
from model import opt

os.makedirs("image", exist_ok=True)

adversarial_loss = nn.BCELoss()

# generator = model.Generator()
# discriminator = model.Discirminator()
generator = model.CNN_G()
discriminator = model.CNN_D()

os.makedirs('mnist', exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        root='./mnist',
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True
)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        valid = Tensor(imgs.size(0), 1).fill_(1.0)
        valid.requires_grad = False
        fake = Tensor(imgs.size(0), 1).fill_(0.0)
        valid.requires_grad = False

        real_imgs = imgs.type(Tensor)

        # train Generator
        optimizer_G.zero_grad()

        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        gen_img = generator(z)
        g_loss = adversarial_loss(discriminator(gen_img), valid)

        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_img.detach()), fake)
        # detach 后gen_img参数不更新
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_img.data[:25], "image/%d.png" % batches_done, nrow=5, normalize=True)

torch.save(discriminator.state_dict(), opt.D_path)
torch.save(generator.state_dict(), opt.G_path)
