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

class BBDiscriminator(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super(BBDiscriminator, self).__init__()
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
    
class BBDiscriminatorW(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super(BBDiscriminatorW, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128), #First hidden layer
            nn.ReLU(),
            nn.Linear(128, 256), #Second hidden layer
            nn.ReLU(), 
            nn.Linear(256, 1),
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

class BiggerDiscriminator(nn.Module): 
    def __init__(self, input_size):
        self.input_size = input_size
        super(BiggerDiscriminator, self).__init__()
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

#Not used, was for work on DCGAN
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)