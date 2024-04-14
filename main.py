from models import BBGenerator2, BBDescriminatorW
from training_loop import train_linear_model, train_linear_model_w
from data_loader import load_mnist 

G = BBGenerator2(128, 28*28)
D = BBDescriminatorW(28*28)
mnist_dataloader = load_mnist()

train_linear_model_w(G, D, 128, "WGAN_Higher_LRbn", mnist_dataloader)