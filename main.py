import torch
from models import * 
from helper_functions import convert_mnist_image_to_output

G = MNIST_DCGAN_G(64)
noise = torch.randn(1, 64)

output = G(noise)[0]
print(len(output))
print(output.shape)
print(len(output[0]))
print(len(output[0].view(-1)))

img = convert_mnist_image_to_output(output[0].view(-1))
D = MNIST_DCGAN_D()
classification = D(output)

print(classification)

print(classification.shape)