import torch
BATCH_SIZE = 64

if torch.cuda.is_available():
    # Move the model to CUDA
    device = torch.device("cuda")
