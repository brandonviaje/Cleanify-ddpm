import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Load CIFAR10 Dataset
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

# DataLoader 
train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=64, shuffle=False)
