import numpy as np
import torch
from torch import nn, optim
from unet import Unet
from descriminator import PatchDiscriminator
from gan_loss import GANLoss
from weight_management import init_model
from skimage.color import lab2rgb


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def lab_to_rgb(self, L, ab):
        """
        Takes a batch of images
        """

        L = (L + 1.) * 50.
        ab = ab * 110.
        Lab = torch.cat([L, ab], dim=1)
        L2 = Lab.permute(0, 2, 3, 1).detach().cpu().numpy()
        rgb_imgs = []
        for img in L2:
            if img.shape[2] == 2:
                img = np.concatenate([img, np.zeros((256, 256, 1))], axis=2)
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)

    def predict(self, PIL_image):
        self.net_G.eval()
        with torch.no_grad():
            self.setup_input({'L': PIL_image, 'ab': torch.zeros_like(PIL_image)})
            self.forward()
        fake_color = self.fake_color.detach()
        real_color = self.ab
        L = self.L
        fake_imgs = self.lab_to_rgb(L, fake_color)
        real_imgs = self.lab_to_rgb(L, real_color)
        return np.concatenate([real_imgs[0], fake_imgs[0]], axis=1)
