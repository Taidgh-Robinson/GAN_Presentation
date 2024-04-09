import torch
from models import *

BATCH_SIZE = 128
NUM_EPOCH = 250

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")


