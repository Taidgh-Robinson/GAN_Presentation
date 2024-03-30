import torch
import librosa
import os
import numpy as np
from torchvision import datasets, transforms
from variables import BATCH_SIZE
from variables import RGB_MEAN, RGB_STD

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,)),
])

mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)
mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

CIFAR10_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)  # Normalize
])


CIFAR_dataset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=CIFAR10_transform)

cat_indices = [i for i, (_, label) in enumerate(CIFAR_dataset) if label == 3]
cat_subset = torch.utils.data.Subset(CIFAR_dataset, cat_indices)

def load_all_birdcalls():
    directory = "data/birdcalls/raw"
    files = os.listdir(directory)
    wavs = []
    for f in files: 
        call, _ = librosa.load(directory+"/"+str(f), duration=3)
        wavs.append(call)
    return np.array(wavs)

calls = load_all_birdcalls()

for c in calls: 
    print(len(c))