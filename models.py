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
    
class BiggerGenerator(nn.Module):
    def __init__(self, noise_size, output_size):
        self.noise_size = noise_size
        self.output_size = output_size
        super(BiggerGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128), #One hidden layer
            nn.ReLU(),
            nn.Linear(128, 256), #Second hidden layer
            nn.ReLU(),
            nn.Linear(256, 512), #Third hidden layer
            nn.ReLU(),
            nn.Linear(512, self.output_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)


class BiggerDiscriminator(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super(BiggerDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512), #One hidden layer
            nn.ReLU(), 
            nn.Linear(512, 256), #Second hidden layer
            nn.ReLU(), 
            nn.Linear(256, 64), #Third hidden layer
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)