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
        #     nn.BatchNorm2d(3),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(3, 32, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(32, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, 3, stride=1, padding=1),
        # )

        # self.upsample_layers = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(32, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(32, 3, 3, stride=1, padding=1),
        #     nn.Tanh()
        # )

        self.downsample_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
        )

        self.upsample_layers = nn.Sequential(
            nn.Conv2d(3, 3, 1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, input_image):
        # latent = self.downsample_layers(input_image)
        out = self.upsample_layers(input_image)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=8,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                    ),                      # output shape (16, 28, 28)
            nn.LeakyReLU(0.2),              # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (8, 14, 14)
            
            nn.Conv2d(
                in_channels=8,             # input height
                out_channels=64,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                    ),                      # output shape (32, 14, 14)
            nn.LeakyReLU(0.2),              # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 7, 7)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(60000,1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # out = self.fc_layers(conv_output.view(input_image.shape[0],-1))
        combined = torch.cat([x, y], dim=2).view(x.shape[0], -1)
        out = self.fc_layers(combined)
        return out
        
def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    if args.summary:
        # Tensorboard writer
        writer = SummaryWriter(os.path.join('output', 'gan', 'summary', args.model_name))

    adversarial_loss = torch.nn.BCELoss().cuda()

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

            # print(discriminator(g, imgs))

            g_loss = a_loss #+ l1_loss
            
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
                    "G_loss = {:.3f}, D_loss = {:.3f}".format(
                        epoch, i,
                        args.batch_size, g_loss, d_loss
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
    parser.add_argument('--lr_G', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_D', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='amount of examples in the minibatch')

    parser.add_argument('--model_name', type=str, default='gantest', help='dir to save stuff')       
    parser.add_argument('--save_interval', type=int, default=100, help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--summary', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help='Make summary')

    args = parser.parse_args()
    main()