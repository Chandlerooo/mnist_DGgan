import os
import torch
from torchvision.utils import save_image
import model
import numpy as np
from model import opt

np.random.seed(0)
net = model.Generator()
net.load_state_dict(torch.load(opt.G_path))
net.eval()

x = np.random.normal(0, 1, (1, opt.latent_dim - 2))
x = np.tile(x, (64, 1))
y_1 = np.linspace(-1, 1, 8).reshape(-1, 1)
y = np.array([[*i, *j] for j in y_1 for i in y_1])
# y = np.random.normal(0, 1, (64, 1))
for N in range(64):
    z = np.concatenate((x[:, :N], y, x[:, N:]), 1)
    z = torch.FloatTensor(z)

    gen_img = net(z)

    os.makedirs('image_v', exist_ok=True)
    save_image(gen_img.data, "image_v/sample_%d.png" % N, nrow=8, normalize=True)
