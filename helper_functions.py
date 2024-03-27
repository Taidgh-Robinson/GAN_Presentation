import torch
import numpy as np
from PIL import Image
from main import device

def convertOutputToImage(generatorOutput):
    output = generatorOutput.cpu().detach().numpy().reshape(28,28)
    output = (output + 1) / 2
    output *= 255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    return image
    
def gen_images(G, number_of_images, shape_of_noise):
    if(len(shape_of_noise) == 1 ):
        noise = torch.randn(number_of_images, shape_of_noise[0])
    elif(len(shape_of_noise) == 2):
        noise = torch.randn(number_of_images, 1, shape_of_noise[0], shape_of_noise[1])
    images = []
    with torch.no_grad():
        fake_images = G(noise.to(device))
        for image in fake_images:
            img = convertOutputToImage(image)
            images.append(img)
    return images