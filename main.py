from models import BBGenerator2, BBDiscriminatorW
from training_loop import train_linear_model, train_linear_model_w
from data_loader import load_mnist 

G = BBGenerator2(32, 28*28)
D = BBDiscriminatorW(28*28)
mnist_dataloader = load_mnist()

train_linear_model_w(G, D, 32, "WGAN_Higher_LRsn2", mnist_dataloader)