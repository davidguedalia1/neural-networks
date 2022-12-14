import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch import Tensor
from torch.autograd import Variable
import pandas
from torchvision import models
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

batch_size = 128
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def mnist_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                             shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)

    return dataloader, testloader


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def train_gan(num_epochs, netD, netG, optimizerD, optimizerG,loss_function):
    real_label = 1.
    fake_label = 0.
    ngpu = 1
    nc = 1
    loss_generator = []
    loss_discriminator = []
    img_list = []
    dataloader, testloader = mnist_data()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    iters = 0
    k = 3
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            for _ in range(k):
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_cpu).view(-1, 1).squeeze(1)
                errD_real = loss_function(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1, 1).squeeze(1)
                errD_fake = loss_function(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake).view(-1, 1).squeeze(1)
            errG = loss_function(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            loss_generator.append(errG.item())
            loss_discriminator.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    plt.plot(loss_discriminator, label="discriminator loss")
    plt.plot(loss_generator, label="generator loss")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    plt.axis("off")
    plt.show()

lr = 0.0002
beta1 = 0.5
nz = 10
netD_bce = Discriminator(ngpu, 32, 1).to(device)
netG_bce = Generator(ngpu, nz, 32, 1).to(device)
optimizerD_bce = optim.Adam(netD_bce.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG_bce = optim.Adam(netG_bce.parameters(), lr=lr, betas=(beta1, 0.999))

class SaturatedBCELoss(nn.BCELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return -super(SaturatedBCELoss, self).forward(input, 1 - target)

class SaturatedBCELoss(nn.BCELoss):
    def forward(self, input: Tensor, target: Tensor):
        return -super(SaturatedBCELoss, self).forward(input, 1 - target)

def q1_bce_loss():
    train_gan(50, netD_bce, netG_bce, optimizerD_bce, optimizerG_bce, nn.BCELoss())

q1_bce_loss()

lr = 0.0002
beta1 = 0.5
nz = 10
netD_mse = Discriminator(ngpu, 32, 1).to(device)
netG_mse = Generator(ngpu, nz, 32, 1).to(device)
optimizerD_mse = optim.Adam(netD_mse.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG_mse = optim.Adam(netG_mse.parameters(), lr=lr, betas=(beta1, 0.999))

def q1_mse_loss():
    train_gan(20, netD_mse, netG_mse, optimizerD_mse, optimizerG_mse, nn.MSELoss())

q1_mse_loss()

lr = 0.0002
beta1 = 0.5
nz = 10
netD_bce_stat = Discriminator(ngpu, 32, 1).to(device)
netG_bce_stat = Generator(ngpu, nz, 32, 1).to(device)
optimizerD_bce_stat = optim.Adam(netD_bce_stat.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG_bce_stat = optim.Adam(netG_bce_stat.parameters(), lr=lr, betas=(beta1, 0.999))

def q1_bce_stat_loss():
    train_gan(5, netD_bce_stat, netG_bce_stat, optimizerD_bce_stat, optimizerG_bce_stat, SaturatedBCELoss())

q1_bce_stat_loss()


def show_images(im1, im2):
    for i in range(5):
        plt.figure()
        img = transforms.ToPILImage()(im2[i])
        plt.title("The original image")
        plt.imshow(img)
        img = transforms.ToPILImage()(im1[i])
        plt.title("The generated image")
        plt.imshow(img)

def z_from_image(im, gen, lr, max_epoch):
  gen.eval()
  images = []
  z = torch.randn(im.size(0), 10, 1, 1, device=device, requires_grad=True)
  loss_function = nn.MSELoss()
  optimizer = optim.Adam([z], lr)
  for i in range(max_epoch):
    optimizer.zero_grad()
    G_z = gen(z)
    loss = loss_function(im.view((1, 1, 28, 28)).detach(), G_z)
    loss.backward()
    optimizer.step() 
  output = gen(z)
  show_images(output, im)
  return z

transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize((0.5), (0.5))])

dataset = torchvision.datasets.MNIST(root='./data', 
                                     download=True, transform=transform)

testloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                          shuffle=False, num_workers=0)

def question2():
    iter_data = iter(testloader)
    for i in range(5):
        im, _ = next(iter_data)
        z_from_im(im.to(device), netG_bce, 0.0002, 50000)


transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 7)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def normal_distirbution(z):
  kurtos = (t-torch.mean(z))/torch.std(z)
  kurtos = torch.mean(torch.pow(kurtos, 4.0))
  return torch.pow(torch.mean(z), 2) + torch.pow((torch.std(z)-1),2) + torch.pow((kurtos-3),2)

def plot_images(images):
  toPIL = transforms.ToPILImage()
  imgs_len = len(images)
  plt.figure(figsize=(imgs_len, 1))
  for i, (title, image) in enumerate(images):
    img = toPIL(image[0])
    ax1 = plt.subplot(1, imgs_len, i+1)
    ax1.title.set_text(f"{title}")
    plt.imshow(img)

  plt.show()

def train_ae(model, optimizer, epochs): 
  criterion = nn.MSELoss()
  for epoch in range(epochs):
    model.train()
    for i, data in enumerate(dataloader, 0):
      inputs = data[0].to(device)
      labels = data[1].to(device)
      optimizer.zero_grad()
      z_e =  model.encoder(inputs)
      recon = model.decoder(z_e)
      loss = criterion(recon, inputs) + normal_distirbution(z_e)
      loss.backward()
      optimizer.step()

def show_images(gen):
  gen.eval()
  with torch.no_grad():
    noise = torch.randn(3, 10, 1, 1, device=device)
    to_visual = gen(noise).cpu()
    for im in to_visual:
      plt.figure()
      plt.imshow(transforms.ToPILImage()(im.view(28,28)))
    plt.gray()
    plt.show()

model_new = Autoencoder().to(device)
train_ae(model_new, dataloader)

wae = "data/wae.pth"
torch.save(model_new.state_dict(), wae)

l_net = Autoencoder().to(device)
l_net.load_state_dict(torch.load(w_ae_save_name))

show_images(l_net)

def question4(l_net):
    dataloader = mnist_data()[1]
    img, _ = next(iter(dataloader))
    first_image = img[0].reshape(1, 1, 28, 28)
    second_image = img[1].reshape(1, 1, 28, 28)
    z_one_1 = l_net.encoder(first_image.to(device))
    z_two_1 = l_net.encoder(second_image.to(device))
    for i, a in enumerate(np.linspace(0, 1, 15)):
        fig, axs = plt.subplots(3, 5, i+1)
        output = (z_one_1 * a + (1 - a) * z_two_1).reshape(1, 10, 1, 1)
        output = l_net.decoder(output).detach().cpu()
        plt.imshow(output.reshape(28, 28))
        plt.title(f"a={str(a)[:3]}")
        plt.gray()
        plt.show()
    plt.show()