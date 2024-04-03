import torch
from models import * 
from helper_functions import convert_mnist_image_to_output
from training_loop import * 
from data_loader import * 


noise = torch.randn(3, 64)
mnist = load_mnist()


G = MNIST_DCGAN_G(64)
D = MNIST_DCGAN_D()
output = G(noise)
print(output)
classifcation = D(output)
print(classifcation)
print(classifcation.view(classifcation.size(0), -1).shape)

train_convolutional_model(G, D, 64, "adsfafd", mnist)