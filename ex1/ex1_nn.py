"""
Neural Networks for Images - Ex1
explore and evaluate the importance of various components of a classification CNN.
This will be done by adding, modifying and removing different components.
In each of these experiments you will need to implement the change, train the network.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

batch_size = 4
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self, conv1_out=20, conv_layer=5, fc_layer=120):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, conv_layer)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, 16, conv_layer)

        self.fc1 = nn.Linear(16 * 5 * 5, fc_layer)
        self.fc2 = nn.Linear(fc_layer, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, num_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_losses_by_epoch = []
    test_losses_by_epoch = []
    for epoch in range(num_epoch):  # loop over the dataset multiple times
      running_loss = 0.0
      epoch_running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          # zero the parameter gradients
          optimizer.zero_grad()
          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          epoch_running_loss += loss.item()

          if i % 2000 == 1999:    # print every 2000 mini-batches
              print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
              running_loss = 0.0
      train_loss, train_accuracy = get_loss(trainloader, net)
      test_loss, test_accuracy = get_loss(testloader, net)
      train_losses_by_epoch.append(train_loss)
      test_losses_by_epoch.append(test_loss)
    print('Finished Training')
    return train_losses_by_epoch, test_losses_by_epoch


def get_loss(data_loader, net):
    criterion = nn.CrossEntropyLoss()
    c = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            c += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    average_loss = running_loss / len(data_loader)
    accuracy = 100 * c / total
    print(accuracy)
    return average_loss, accuracy


def plot_filters_losses(n, loss_train, loss_test):
    plt.plot(n, loss_train, 'r', label='Train loss')
    plt.plot(n, loss_test, 'g', label='Test loss')
    plt.title('Training loss and Test loss')
    plt.xlabel('Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run_architecture_models(type_net):
    train_losses, test_losses = [], []
    list_epoch = [1, 2, 3, 4, 5]
    filters_list = [2, 5, 10, 15, 20, 25, 30, 35]
    for i in filters_list:
        net = type_net(conv1_out=i)
        train_epoch, test_epoch = train(net, 5)
        plot_filters_losses(list_epoch, train_epoch, test_epoch)
        train_loss, train_accuracy = get_loss(trainloader, net)
        test_loss, test_accuracy = get_loss(testloader, net)  
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    plot_filters_losses(filters_list, train_losses, test_losses)


class LinearNet(nn.Module):
    def __init__(self, conv1_out=6, conv_layer=5, fc_layer=120):
        super(LinearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, conv_layer)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, 16, conv_layer)

        self.fc1 = nn.Linear(16 * 5 * 5, fc_layer)
        self.fc2 = nn.Linear(fc_layer, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        x =  self.pool(self.conv1(x))
        x =  self.pool(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def run_linear_models(type_net):
    train_losses, test_losses = [], []
    filters_list = [25, 100]
    for i in filters_list:
        net = type_net(conv1_out=i)
        train(net, 2)
        train_loss, train_accuracy = get_loss(trainloader, net)
        test_loss, test_accuracy = get_loss(testloader, net)  
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    plot_filters_losses(filters_list, train_losses, test_losses)


class ShallowNetV1(nn.Module):
    def __init__(self, conv1_out=20, conv_layer=5, fc_layer=120):
        super(ShallowNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, conv_layer)
        self.fc1 = nn.Linear(conv1_out * 28 * 28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x



if __name__ == '__main__':
    # run_architecture_models(Net)
    # run_linear_models(LinearNet)
    # run_architecture_models(ShallowNetV1)

