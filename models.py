import torch
import torch.nn as nn


class BBGenerator(nn.Module): 
    def __init__(self, noise_size, output_size):
        self.noise_size = noise_size
        self.output_size = output_size
        super(BBGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 128),
            nn.Tanh(),
            nn.Linear(128, 512), #One hidden layer
            nn.Tanh(),
            nn.Linear(512, self.output_size),
            nn.Tanh()
        )

    def forward(self, x): 
        return self.model(x)

class BBDescriminator(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super(BBDescriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),  #One hidden layer
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class MNIST_DCGAN_G(nn.Module):
    def __init__(self, noise_size):
        self.noise_size = noise_size
        super(MNIST_DCGAN_G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 256*7*7), #Project
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),  # Reshape
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)
    
class MNIST_DCGAN_D(nn.Module):
    def __init__(self):
        super(MNIST_DCGAN_D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=3, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        return self.model(x)

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)