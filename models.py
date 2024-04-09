import torch
import torch.nn as nn

class BBGenerator(nn.Module): 
    def __init__(self, noise_size, output_size):
        self.noise_size = noise_size
        self.output_size = output_size
        super(BBGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 128), #First hidden layer
            nn.Tanh(),
            nn.Linear(128, 512), #Second hidden layer
            nn.Tanh(),
            nn.Linear(512, self.output_size), #Output layer
            nn.Tanh()
        )

    def forward(self, x): 
        return self.model(x)

class BBDescriminator(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super(BBDescriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128), #First hidden layer
            nn.ReLU(),
            nn.Linear(128, 256), #Second hidden layer
            nn.ReLU(), 
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
class BBGenerator2(nn.Module): 
    def __init__(self, noise_size, output_size):
        self.noise_size = noise_size
        self.output_size = output_size
        super(BBGenerator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 128), #First hidden layer
            nn.ReLU(),
            nn.Linear(128, 512), #Second hidden layer
            nn.ReLU(),
            nn.Linear(512, self.output_size), #Output layer
            nn.Tanh()
        )

    def forward(self, x): 
        return self.model(x)
    
class BiggerGenerator(nn.Module): 
    def __init__(self, noise_size, output_size):
        self.noise_size = noise_size
        self.output_size = output_size
        super(BiggerGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 64), #First hidden layer
            nn.ReLU(), 
            nn.Linear(64, 128), #Second hidden layer
            nn.ReLU(), 
            nn.Linear(128, 128), #Third hidden layer
            nn.ReLU(), 
            nn.Linear(128, 256), #Fourth hidden layer
            nn.ReLU(), 
            nn.Linear(256, 512), #Fifth hidden layer
            nn.ReLU(), 
            nn.Linear(512, self.output_size), 
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class BiggerDescriminator(nn.Module): 
    def __init__(self, input_size):
        self.input_size = input_size
        super(BiggerDescriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 512), #First hidden layer
            nn.ReLU(), 
            nn.Linear(512, 512), #Second hidden layer
            nn.ReLU(), 
            nn.Linear(512, 256), #Third hidden layer
            nn.ReLU(), 
            nn.Linear(256, 128), #Fourth hidden layer
            nn.ReLU(), 
            nn.Linear(128, 64), #Fifth hidden layer
            nn.ReLU(), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class MNIST_DCGAN_G(nn.Module):
    def __init__(self, noise_size):
        self.noise_size = noise_size
        super(MNIST_DCGAN_G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 128*7*7), #Project
            nn.Unflatten(1, (128, 7, 7)),  # Reshape
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.model(x)
    
class MNIST_DCGAN_D(nn.Module):
    def __init__(self):
        super(MNIST_DCGAN_D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        return self.model(x)


class DiscriminatorGPT(nn.Module):
    def __init__(self):
        super(DiscriminatorGPT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*7*7, 1)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2_bn(self.conv2(x)))
        x = x.view(-1, 128*7*7)
        x = torch.sigmoid(self.fc(x))
        return x

# Define the generator network
class GeneratorGPT(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorGPT, self).__init__()
        self.fc = nn.Linear(z_dim, 128*7*7)
        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        x = torch.relu(self.convT1(x))
        x = torch.tanh(self.convT2(x))
        return x


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)