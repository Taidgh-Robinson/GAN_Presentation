import torch
import numpy as np
from PIL import Image
from variables import device
import os
from torchvision import transforms
from variables import RGB_MEAN, RGB_STD

def convert_mnist_image_to_output(generator_output):
    output = generator_output.cpu().detach().numpy().reshape(28,28)
    output = (output + 1) / 2
    output *= 255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    return image


def convert_CIFAR_image_to_output(preprocessed_image):
    inverse_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(RGB_MEAN, RGB_STD)],
    std=[1/s for s in RGB_STD]
    )

    denormalized_image = inverse_normalize(preprocessed_image)
    image = transforms.ToPILImage()(denormalized_image).convert("RGB")    
    return image

def gen_images(G, number_of_images, shape_of_noise):
    noise = torch.randn(number_of_images, shape_of_noise)
    images = []
    with torch.no_grad():
        fake_images = G(noise.to(device))
        for image in fake_images:
            img = convert_mnist_image_to_output(image)
            images.append(img)
    return images

def gen_sound(G, number_of_clips, shape_of_noise): 
    noise = torch.randn(number_of_clips, shape_of_noise)
    clips = []
    with torch.no_grad():
        fake_clips = G(noise.to(device))
        for clip in fake_clips:
            wav = convert_output_to_wav(clip)
            clips.append(wav)
    return clips


def convert_output_to_wav(generator_output):
    output = generator_output.cpu().detach().numpy()
    sample_rate = 22050
    return np.int16(output * 32767)

def create_gif_from_images(image_folder, gif_path):    
    image_tags = [i*10 for i in range(50)]
    image_filenames = [str(i)+'.jpg' for i in image_tags]
    # Create a list to store image objects
    images = []
    
    # Open each image and append it to the list
    for filename in image_filenames:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        images.append(image)
    
    # Save the images as a GIF
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)