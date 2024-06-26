import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import weights_init
from helper_functions import gen_images
from variables import device, NUM_EPOCH, N_DISCRIMINATOR
from loss_functions import wasserstein_loss

def train_linear_model(G, D, shape_of_noise, name_of_model, train_loader):
    G.to(device)
    D.to(device)
    loss = nn.BCELoss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr = 0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr = 0.0001)
    
    output_directory = "data/generated_images/"+str(name_of_model)
    if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    for epoch in range(NUM_EPOCH):
        for i, (real_images, _) in enumerate(train_loader):
            
            size_of_current_batch = real_images.size(0)

            #Generate noise based on the input shape of the model we are training            
            noise = torch.randn(size_of_current_batch, shape_of_noise)

            #Generate some fake "images"
            fake_data = G(noise.to(device))
            #Actual mnist pictures
            real_data = real_images.view(real_images.size(0), -1).to(device)
            #Labels for loss, 1 = MNIST, 0 = Generated 
            real_labels = torch.ones(size_of_current_batch, 1).float().to(device)
            fake_labels = torch.zeros(size_of_current_batch, 1).float().to(device)

            optimizer_D.zero_grad()
            #How wrong was our Discriminator on the MNIST images
            real_loss = loss(D(real_data), real_labels)
            #How wrong was our Discriminator on the Generated images
            fake_loss = loss(D(fake_data.detach()), fake_labels)
            #Add them to find out how wrong it was in total
            total_loss = real_loss + fake_loss

            #Back propegate that loss
            total_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            #Generate the fake data again for new testing (IIRC this is pytorch-fu)
            fake_data = G(noise.to(device))
            #What does our more trained Discriminator think about these images
            fake_output = D(fake_data.to(device))
            #Loss for the generator is what datapoints did it not predict 1 on (i.e. if it predictied 100% 1's our loss would be 0)
            g_loss = loss(fake_output, real_labels)

            #Back propegate that loss
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(
                f"Epoch [{epoch}/{NUM_EPOCH}], Batch {i}/{len(train_loader)}, "
                f"Discriminator Loss: {total_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}"
                )

        #Every 10 epochs lets save some generated images to see what our models think
        if(epoch % 10 == 0):
            images = gen_images(G, 1, shape_of_noise, False)
            for i, image in enumerate(images):
                image_path = os.path.join(output_directory, str(epoch)+".jpg")
                image.save(image_path)

    torch.save(G.state_dict(), "output/G" + str(name_of_model) + ".pth")
    torch.save(D.state_dict(), "output/D" + str(name_of_model) + ".pth")

def train_linear_model_w(G, D, size_of_noise, name_of_model, train_loader):
    G.to(device)
    D.to(device)
    optimizer_G = optim.Adam(G.parameters(), lr = 0.00007)
    optimizer_D = optim.Adam(D.parameters(), lr = 0.00001)
    
    output_directory = "data/generated_images/"+str(name_of_model)
    if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    for epoch in range(NUM_EPOCH):
        for i, (real_images, _) in enumerate(train_loader): 
            batch_size = real_images.size(0)

            for _ in range(N_DISCRIMINATOR):
                noise = torch.randn(batch_size, size_of_noise).to(device)
                fake_data = G(noise)
                real_data = real_images.view(real_images.size(0), -1).to(device)
                
                optimizer_D.zero_grad()
                d_real = D(real_data)
                d_fake = D(fake_data.detach())

                d_loss = wasserstein_loss(d_real, d_fake)
                d_loss.backward()
                #clip weights
                for param in D.parameters():
                    param.data.clamp_(-0.01, 0.01)

                optimizer_D.step()


            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, size_of_noise).to(device)
            fake_data = G(noise)
            fake_score = D(fake_data)
            g_loss = -1. * torch.mean(fake_score)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(
                f"Epoch [{epoch}/{NUM_EPOCH}], Batch {i}/{len(train_loader)}, "
                f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}"
                )
        
        if(epoch % 10 == 0):
            images = gen_images(G, 1, size_of_noise, False)
            for i, image in enumerate(images):
                image_path = os.path.join(output_directory, str(epoch)+".jpg")
                image.save(image_path)

    torch.save(G.state_dict(), "output/G" + str(name_of_model) + ".pth")
    torch.save(D.state_dict(), "output/D" + str(name_of_model) + ".pth")

#Not used, was for work on DCGAN
def train_convolutional_model(G, D, size_of_noise, name_of_model, train_loader):
    G.to(device)
    D.to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    loss = nn.BCELoss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    output_directory = "data/generated_images/"+str(name_of_model)
    if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    for epoch in range(NUM_EPOCH): 
        for i, (real_images, _) in enumerate(train_loader):

            size_of_current_batch = real_images.size(0)
            noise = torch.randn(size_of_current_batch, size_of_noise)
            fake_data = G(noise.to(device))
            real_data = real_images.to(device)
            
            real_labels = torch.ones(size_of_current_batch, 1).float().to(device)
            fake_labels = torch.zeros(size_of_current_batch, 1).float().to(device)
            
            optimizer_D.zero_grad()
            #How wrong was our Discriminator on the MNIST images
            classification_real = D(real_data)
            real_loss = loss(classification_real.view(classification_real.size(0), -1), real_labels)
            #How wrong was our Discriminator on the Generated images
            classification_fake = D(fake_data.detach())
            fake_loss = loss(classification_fake.view(classification_fake.size(0), -1), fake_labels)
            #Add them to find out how wrong it was in total
            total_loss = real_loss + fake_loss

            #Back propegate that loss
            total_loss.backward()
            optimizer_D.step()

            fake_data = G(noise.to(device))
            #What does our more trained Discriminator think about these images
            fake_output = D(fake_data.to(device))
            #Loss for the generator is what datapoints did it not predict 1 on (i.e. if it predictied 100% 1's our loss would be 0)
            g_loss = loss(fake_output.view(fake_output.size(0), -1), real_labels)

            #Back propegate that loss
            g_loss.backward()
            optimizer_G.step()
            if i % 100 == 0:
                print(
                f"Epoch [{epoch}/{NUM_EPOCH}], Batch {i}/{len(train_loader)}, "
                f"Discriminator Loss: {total_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}"
                )
        #Every 10 epochs lets save some generated images to see what our models think
        if(epoch % 10 == 0):
            images = gen_images(G, 1, size_of_noise, True)
            for i, image in enumerate(images):
                image_path = os.path.join(output_directory, str(epoch)+".jpg")
                image.save(image_path)

    torch.save(G.state_dict(), "output/G" + str(name_of_model) + ".pth")
    torch.save(D.state_dict(), "output/D" + str(name_of_model) + ".pth")