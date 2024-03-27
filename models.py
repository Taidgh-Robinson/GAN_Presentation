import torch
import torch.nn as nn

class Generator(torch.nn.Module):
    def __init__(self): 
        super(Generator, self).__init__()
        self.convolution = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=0)
        self.linear2 = torch.nn.Linear(25, 64)
        self.linear3 = torch.nn.Linear(64, 128)
        self.linear4 = torch.nn.Linear(128, 784)
        self.activation4 = torch.nn.Tanh() 

    def forward(self, x):
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        x = self.activation4(self.linear2(x))
        x = self.activation4(self.linear3(x))
        x = self.activation4(self.linear4(x))
        return x

class Descriminator(torch.nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, 128)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(128, 1024)
        self.linear3 = torch.nn.Linear(1024, 128)
        self.linear4 = nn.Linear(128, 1)
        self.activation3 = nn.Sigmoid()

    
    def forward(self, x): 
        x = self.leaky(self.linear1(x))
        x = self.dropout(x)
        x = self.leaky(self.linear2(x))
        x = self.dropout(x)
        x = self.leaky(self.linear3(x))
        x = self.dropout(x)

        x = self.activation3(self.linear4(x))

        return x



class BBGenerator(nn.Module): 
    def __init__(self, noise_size):
        self.noise_size = noise_size
        super(BBGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, x): 
        return self.model(x)

class BBDescriminator(nn.Module):
    def __init__(self):
        super(BBDescriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)