import wandb
wandb.init(project="EX2 - Neural Networks for Images", entity="david-guedalia")
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class AEDeep(nn.Module):
  def __init__(self, d=10):
    super(AEDeep, self).__init__()
    self.encoder = Encoder2(d)
    self.decoder = Decoder2(d)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Encoder2(nn.Module):
    def __init__(self, d):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 25, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(25, 60, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(60)
        self.conv3 = nn.Conv2d(60, 100, 3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(100, 128, 3, stride=1, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, d)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder2(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 128)    
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 1, 1))
        self.conv1t = nn.ConvTranspose2d(128, 100, 3, stride=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2t = nn.ConvTranspose2d(100, 60, 3, stride=2, output_padding=0)
        self.bn2 = nn.BatchNorm2d(60)
        self.conv3t = nn.ConvTranspose2d(60, 25, 3, stride=2, padding=1, output_padding=1)
        self.conv4t = nn.ConvTranspose2d(25, 1, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = F.relu(self.bn1(self.conv1t(x)))
        x = F.relu(self.bn2(self.conv2t(x)))
        x = F.relu(self.conv3t(x))
        x = torch.sigmoid(self.conv4t(x))
        return x

class AEHeavy(nn.Module):
  def __init__(self, d=10):
    super(AEHeavy, self).__init__()
    self.encoder = Encoder3(d)
    self.decoder = Decoder3(d)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Encoder3(nn.Module):
    def __init__(self, d):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 25, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(25, 50, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 60, 3, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3 * 3 * 60, 128)
        self.fc2 = nn.Linear(128, d)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder3(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 3 * 3 * 60)    

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(60, 3, 3))
        self.conv1t = nn.ConvTranspose2d(60, 50, 3, stride=2, output_padding=0)
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2t = nn.ConvTranspose2d(50, 25, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(25)
        self.conv3t = nn.ConvTranspose2d(25, 1, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = F.relu(self.bn1(self.conv1t(x)))
        x = F.relu(self.bn2(self.conv2t(x)))
        x = torch.sigmoid(self.conv3t(x))
        return x

class AE(nn.Module):
  def __init__(self, d=10):
    super(AE, self).__init__()
    self.encoder = Encoder(d)
    self.decoder = Decoder(d)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3 * 3 * 32, 128)
        self.fc2 = nn.Linear(128, d)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 3 * 3 * 32)    

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.conv1t = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2t = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3t = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = F.relu(self.bn1(self.conv1t(x)))
        x = F.relu(self.bn2(self.conv2t(x)))
        x = torch.sigmoid(self.conv3t(x))
        return x

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def train(model, device, dataloader, loss_mse, optimizer, loss_score):
    model.train()
    train_loss = []
    for x, y in dataloader:
        x = x.to(device)
        output = model(x)
        loss = loss_mse(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        loss_score["train_loss"] += loss.item() / len(dataloader)
    return np.mean(train_loss)

def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DATA_PATH = '../data_sets/mnist'

    train_set = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True)
    return train_loader, test_loader, train_set, test_set

def test(ae, device, dataloader, loss_fn, loss_score):
    ae.eval()
    with torch.no_grad():
        out = []
        l = []
        for image_batch, i in dataloader:
            image = image_batch.to(device)
            decoded_data = ae(image)
            loss = loss_fn(decoded_data, image)
            out.append(decoded_data.cpu())
            conc_label.append(image.cpu())
            loss_score["test_loss"] += loss.item() / len(dataloader)
        out = torch.cat(out)
        l = torch.cat(l) 
        val_loss = loss_fn(out, l)
    return val_loss.data

def train_test_model(ae, loss_mse, device, optimizer, name_model):
    run = wandb.init(project="EX2 - Neural Networks for Images", entity="david-guedalia", name=name_model)
    num_epochs = 30
    train_loader, test_loader, train_set, test_set = get_data()
    for epoch in range(num_epochs):
      loss_score = dict()
      loss_score["train_loss"] = 0
      loss_score["test_loss"] = 0
      train_loss = train(ae, device, train_loader, loss_mse, optimizer, loss_score)
      val_loss = test(ae,device,test_loader,loss_mse, loss_score)
      wandb.log(loss_score)
    run.finish()

list_spaces = [2, 5, 10, 15, 20, 25, 30]
for l in list_spaces:
    loss_mse = torch.nn.MSELoss()
    ae = AE(l)
    learning_rate = 0.001
    device = get_device()
    ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
    train_test_model(ae, loss_mse, device, optimizer,  f"Q1 - latent_space{l}")
    torch.save(ae.encoder, f"Encoder_{l}.pth")
    torch.save(ae.decoder, f"Decoder_{l}.pth")

loss_mse = torch.nn.MSELoss()
ae = AE()
learning_rate = 0.001
device = get_device()
ae.to(device)
optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
train_test_model(ae, loss_mse, device, optimizer,  f"Q1 - latent_space {l} - AE-Main")
torch.save(ae.encoder, f"Encoder_{l}.pth")
torch.save(ae.decoder, f"Decoder_{l}.pth")

loss_mse = torch.nn.MSELoss()
ae_full = AEDeep(20)
learning_rate = 0.001
device = get_device()
ae_full.to(device)
optimizer = optim.Adam(ae_full.parameters(), lr=learning_rate)
train_test_model(ae_full, loss_mse, device, optimizer,  f"Q1 - latent_space 20 - AE- Deep")
torch.save(ae_full.encoder, f"Encoder_Deep_20.pth")
torch.save(ae_full.decoder, f"Decoder_Deep_20.pth")

def display(images):
    for index, image in enumerate(images):
        ax = plt.subplot(3, 5, index + 1)
        plt.imshow(image[0].detach().cpu().numpy().reshape(28, 28), cmap="gray")
        plt.title(f"a={image[1]:.1f}")
    plt.show()

def calc_interpolate(loader, decoder, encoder):
    result = [] 
    iter_loader = iter(loader)
    s = next(iter_loader)
    first_im = s[0][0][:, None]
    second_im = s[0][1][:, None]
    range_alphas = np.linspace(0, 1, 15)
    alpha = 0.1
    device = get_device()
    for a in range_alphas:
        a_encoder = a * encoder(first_im.to(device)) + (1 - alpha) * encoder(second_im.to(device))
        res = decoder(a_encoder)
        result.append((res, a))
    return result

def interpolate_AE():
    DATA_PATH = '../data_sets/mnist'
    decoder = torch.load("Decoder_Deep_20.pth")
    encoder = torch.load("Encoder_Deep_20.pth")
    transf_normal = transforms.Normalize((0.5,), (0.5,))
    transform = transforms.Compose([transforms.ToTensor(), transf_normal])
    mnist_data = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(mnist_data, batch_size=2, shuffle=True)
    images = calc_interpolate(loader, decoder, encoder)
    display(images)

interpolate_AE()

def get_train_data(size, train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DATA_PATH = '../data_sets/mnist'

    train_set = datasets.MNIST(root=DATA_PATH, train=train, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=size, shuffle=True)
    print(len(train_loader))
    return train_loader

def for_each_data(data, res, model):
    for images, label in data:
        ae_image = model(images.cuda())
        for image in ae_image:
            res.extend(image.cpu().numpy())
    im = np.array(res)
    return im.transpose()

def calc_max_corr_matrix(im):
    matrix_corr = np.corrcoef(im) - np.eye(im.shape[0])
    matrix_corr_abs = np.abs(matrix_corr) 
    return matrix_corr_abs.max()

def q3():
    list_spaces = [2, 5, 10, 15, 20, 25, 30]
    data = get_train_data(18, True)
    max_matrix_corr = []
    for l in list_spaces:
        res = []
        model_encoder = torch.load(f"Encoder_{l}.pth")
        model_encoder.requires_grad_(False)
        im = for_each_data(data, res, model_encoder)
        max_matrix_corr.append(calc_max_corr_matrix(im))
    plt.plot(list_spaces, max_matrix_corr)
    plt.xlabel("d - latent space")
    plt.ylabel("Max absolute correlation matrix")
    plt.title("Max absolute of correlation matrix")
    plt.show()

def train_test_model_q4(ae, loss, device, optimizer, name_model, train_loader, test_loader):
    run = wandb.init(project="Ex2-Q4", entity="david-guedalia", name=name_model)
    num_epochs = 30
    for epoch in range(num_epochs):
      loss_score = dict()
      loss_score["train_loss"] = 0
      loss_score["test_loss"] = 0
      train_loss = train4(ae, device, train_loader, loss, optimizer, loss_score)
      val_loss = test(ae,device,test_loader,loss, loss_score)
      wandb.log(loss_score)
    run.finish()

def get_train_data_q4(size, m, train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DATA_PATH = '../data_sets/mnist'
    set_mnist = datasets.MNIST(root=DATA_PATH, train=train, download=True, transform=transform)
    if size == -1:
      data_loader = DataLoader(set_mnist, batch_size=len(set_mnist), shuffle=True)
    else:
      data_loader = DataLoader(set_mnist, batch_size=size, shuffle=True)
    print(len(data_loader))
    return data_loader

def train4(model, device, dataloader, loss_mse, optimizer, loss_score):
    model.train()
    train_loss = []
    for x, y in dataloader:
        x = x.to(device)
        output = model(x)
        loss = loss_mse(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        loss_score["train_loss"] += loss.item() / len(dataloader)
    return np.mean(train_loss)

def q4():
    list_spaces = [2, 5, 10, 15, 20, 25, 30]
    for l in list_spaces:
        train_loader = get_train_data_q4(-1, 80, True)
        test_loader = get_train_data_q4(-1, 80, False)
        model = torch.load(f"Encoder_{l}.pth")
        model.requires_grad_(True)
        model = nn.Sequential(model, nn.Linear(l, 100), nn.ReLU(True),
                                nn.Linear(100, 50), nn.ReLU(True), nn.Linear(50, 10))
        loss = torch.nn.CrossEntropyLoss()
        learning_rate = 0.001
        device = get_device()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_test_model_q4(model, loss, device, optimizer, f"Q4 - latent_space {l} train all", train_loader, test_loader)
        

        train_loader_mlp = get_train_data_q4(80, 80, True)
        test_loader_mlp = get_train_data_q4(80, 80, False)
        model_mlp = torch.load(f"Encoder_{l}.pth")
        model_mlp.requires_grad_(False)
        model_mlp = nn.Sequential(model_mlp, nn.Linear(l, 100), nn.ReLU(True),
                                nn.Linear(100, 50), nn.ReLU(True), nn.Linear(50, 10))
        learning_rate = 0.001
        ae.to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=learning_rate)
        train_test_model_q4(model_mlp, loss, device, optimizer, f"Q4 - latent_space {l} train MLP", train_loader_mlp, test_loader_mlp)