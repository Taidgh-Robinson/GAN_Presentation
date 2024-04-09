from models import * 
from helper_functions import convert_mnist_image_to_output
from training_loop import * 
from data_loader import * 

G = BBGenerator(64, 28*28)
D = BBDescriminator(28*28)
mnist_dataloader = load_mnist()

#train_linear_model(G, D, 64, "BBGANLessEpoch", mnist_dataloader) 
G = GeneratorGPT(100)
D = DiscriminatorGPT()
train_convolutional_model(G, D, 100, "mnistDCGAN", mnist_dataloader)