import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
latent_dim = 100
image_size = 28 * 28
n_epochs = 250
clip_value = 0.01
n_critics = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img)

# Instantiate models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=0.00005)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.00005)

# Training loop
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(trainloader):
        imgs = imgs.view(-1, image_size).to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = -torch.ones(imgs.size(0), 1).to(device)

        # Train discriminator
        for _ in range(n_critics):
            noise = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(noise)

            discriminator.zero_grad()
            real_loss = discriminator(imgs).mean()
            fake_loss = discriminator(fake_imgs.detach()).mean()
            disc_loss = fake_loss - real_loss
            disc_loss.backward()

            # Clip discriminator weights
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            optimizer_D.step()

        # Train generator
        generator.zero_grad()
        noise = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(noise)
        gen_loss = -discriminator(fake_imgs).mean()
        gen_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch: {epoch+1}/{n_epochs}, Batch: {i+1}/{len(trainloader)}, Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")

# Save the trained generator model
torch.save(generator.state_dict(), 'mnist_wgan_g2enerator.pth')