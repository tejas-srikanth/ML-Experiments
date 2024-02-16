import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random

from Discriminator import Discriminator
from Generator import Generator
import torchvision.datasets
import torch.utils.data

import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils



def init_weights(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0, 0.2)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(args, discriminator, generator, dataloader, G_opt, D_opt, device, epoch):
    
    criterion = nn.BCELoss()

    # discriminator_losses = []
    # generator_losses = []
    iters = 0
    real_label = 1
    fake_label = 0
    # grids = []

    # fake_noise = torch.randn(64, 100, 1, 1)

    for i, data in enumerate(dataloader):
        #TRAIN DISCRIMINATOR
        # run real images through discriminator
        images, _ = data
        images = images.to(device)
        b_size = images.size[0]
        D_opt.zero_grad()
        output = discriminator(images).view(-1,)
        labels_real = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
        dreal_loss = criterion(output, labels_real)
        dreal_loss.backward()

        # run fake images through discriminator
        start_vectors = torch.randn(b_size, 100, 1, 1)
        g_output = generator(start_vectors)
        d_output = discriminator(g_output.detach()).view(-1,)
        labels_fake = labels_real.fill_(fake_label)
        dfake_loss = criterion(d_output, labels_fake)
        dfake_loss.backward()

        # optimize discriminator
        # dloss = dfake_loss.item() + dreal_loss.item()
        # discriminator_losses.append(dloss)
        D_opt.step()

        #TRAIN GENERATOR
        G_opt.zero_grad()
        generated_images = generator(start_vectors)
        d_generated_images = discriminator(generated_images).view(-1, 1)
        dgen_loss = criterion(d_generated_images, labels_real)
        # generator_losses.append(dgen_loss.item())
        dgen_loss.backward()
        G_opt.step()
          
        if (i % 50 == 0):
            print(f'EPOCH: {epoch}, i: {i}, Discriminator fake loss: {dfake_loss.item()}, Discriminator real loss: {dreal_loss.item()}, Generator loss: {dgen_loss.item()}')
  

        # if (iters % 500 == 0):
        #     with torch.no_grad():
        #         output = generator(fake_noise).detach()
        #         grids.append(output)
        
        iters += 1

def main():
    batch_size = 128
    nz = 100
    nc = 3
    ngf = 64
    ndf = 64
    image_size = 64
    num_workers = 2
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'],
    #                     help='directory to save the trained model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = dset.ImageFolder(root='./data/',
                            transform=transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    generator = Generator(nz, ngf, nc, 1)

    generator.apply(init_weights)

    discriminator = Discriminator(1, nc, ndf)
    G_opt = optim.Adam(params=generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_opt = optim.Adam(params=discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    discriminator.apply(init_weights)
    for i in range(1, args.epochs+1):
        train(args, discriminator, generator, dataloader, G_opt, D_opt, device, i)
    
    torch.save(discriminator.state_dict(), './models/discriminator.pth')
    torch.save(generator.state_dict(), './models/generator.pth')
    

    