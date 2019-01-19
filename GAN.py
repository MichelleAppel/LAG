import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from LAGDataset import LAGImageFolder

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.downsample_layers = nn.Sequential(
        #     nn.BatchNorm2d(3), # (batch_size, 3, 100, 100)
        #     nn.Conv2d(3, 8, 5, stride=1, padding=2), # (batch_size, 8, 100, 100)
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(0.2, inplace=True), 
        #     nn.MaxPool2d(kernel_size=2), # (batch_size, 8, 50, 50)
        #     nn.Conv2d(8, 16, 5, stride=1, padding=2), # (batch_size, 16, 50, 50)
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=2), # (batch_size, 16, 25, 25)
        #     nn.Conv2d(16, 32, 5, stride=1, padding=2), # (batch_size, 32, 25, 25)
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=2), # (batch_size, 32, 12, 12)
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1), # (batch_size, 64, 12, 12)
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        # self.upsample_layers = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1), # (batch_size, 32, 12, 12)
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2), # (batch_size, 32, 24, 24)
        #     nn.Conv2d(32, 16, 3, stride=1, padding=1), # (batch_size, 16, 24, 24)
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2), # (batch_size, 16, 48, 48)
        #     nn.Conv2d(16, 8, 5, stride=1, padding=3), # (batch_size, 8, 50, 50)
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2), # (batch_size, 8, 50, 50)
        #     nn.Conv2d(8, 3, 3, stride=1, padding=1), # (batch_size, 3, 100, 100)
        #     nn.BatchNorm2d(3),
        #     nn.Tanh()
        # )

        self.down1 = nn.Sequential(
            nn.BatchNorm2d(3), # (batch_size, 3, 100, 100)
            nn.Conv2d(3, 8, 5, stride=1, padding=2), # (batch_size, 8, 100, 100)
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # (batch_size, 8, 50, 50)
            nn.Conv2d(8, 16, 5, stride=1, padding=2), # (batch_size, 16, 50, 50)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # (batch_size, 16, 25, 25)
            nn.Conv2d(16, 32, 3, stride=1, padding=1), # (batch_size, 32, 25, 25)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # (batch_size, 32, 25, 25)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1), # (batch_size, 16, 25, 25)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') # (batch_size, 16, 50, 50)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=1, padding=1), # (batch_size, 8, 50, 50)
            nn.BatchNorm2d(8),
            nn.Upsample(scale_factor=2, mode='bilinear') # (batch_size, 8, 100, 100)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(8, 3, 3, stride=1, padding=1), # (batch_size, 3, 100, 100)
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.sharpen = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=1, padding=1), # (batch_size, 3, 100, 100)
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, input_image):
        # latent = self.downsample_layers(input_image)
        # out = self.upsample_layers(latent)

        h1 = self.down1(input_image)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.middle(h3)
        h5 = self.up1(h4)
        h6 = self.up2(h5)
        out = self.up3(h6)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 1, 1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(100*100, 1),
            nn.Sigmoid()
        )

    def forward(self, target, img):
        d_in = torch.cat((target, img),1)
        hidden = self.conv_layers(d_in)
        validity = self.linear_layers(hidden.view(hidden.shape[0], -1))
        
        return validity

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    if args.summary:
        # Tensorboard writer
        writer = SummaryWriter(os.path.join('output', 'gan', 'summary', args.model_name))

    adversarial_loss = torch.nn.MSELoss().cuda()

    discriminator = discriminator.cuda()
    generator = generator.cuda()

    for epoch in range(args.n_epochs):
        for i, (targets, imgs) in enumerate(dataloader):
            # imgs = imgs.cuda()
            targets = targets.cuda()
            imgs = targets

            # Adversarial ground truths
            valid = torch.Tensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.Tensor(imgs.shape[0], 1).fill_(0.0).cuda()

            g = generator(imgs)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            a_loss = adversarial_loss(discriminator(g, imgs), valid).cuda()
            l1_loss = torch.nn.functional.l1_loss(g, targets).cuda()

            g_loss = l1_loss
            
            # The optimization process
            g_loss.backward() # Perform backward pass
            optimizer_G.step() # Update weights

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(targets, imgs), valid)
            fake_loss = adversarial_loss(discriminator(g.detach(), imgs), fake)

            d_loss = (real_loss + fake_loss)/2

            d_loss.backward() # Perform backward pass
            optimizer_D.step() # Update weights

            # Print progress test
            if i % 10 == 0:
                print("Epoch {}, Train Step {:03d}, Batch Size = {}, "
                    "G_loss = {:.3f}, D_loss = {:.3f} (real loss = {:.3f}, fake loss = {:.3f})".format(
                        epoch, i,
                        args.batch_size, g_loss, d_loss, real_loss, fake_loss
                        )) 

            batches_done = epoch * len(dataloader) + i

            if args.summary:
                writer.add_scalar('g_loss', g_loss, batches_done)
                writer.add_scalar('d_loss', d_loss, batches_done)

            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:

                combined = torch.cat([imgs[:10].view(10, 3, 100, 100), targets[:10].view(10, 3, 100, 100), g[:10].view(10, 3, 100, 100)])
                save_image(combined, os.path.join('output', 'gan', 'images', args.model_name, '{}.png').format(batches_done), nrow=10, normalize=True)

def show_image(image):
    image = image.mul(255).clamp(0, 255).permute(1,2,0).byte().cpu().detach()
    plt.imshow(image)
    plt.show()

def main():
    # create output image directory
    path = os.path.join('output', 'gan')
    os.makedirs(os.path.join(path, 'images', args.model_name), exist_ok=True)

    # load data
    LAGdata = LAGImageFolder('./data/', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(LAGdata, batch_size=args.batch_size, shuffle=True)

    # initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D)

    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr_G', type=float, default=0.02, help='learning rate')
    parser.add_argument('--lr_D', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='amount of examples in the minibatch')

    parser.add_argument('--model_name', type=str, default='gantest', help='dir to save stuff')       
    parser.add_argument('--save_interval', type=int, default=100, help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--summary', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help='Make summary')

    args = parser.parse_args()
    main()