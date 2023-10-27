import torch
import torch.nn as nn
from torchsummary import summary
from PIL import Image
from model import UNet
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

INPUT_RESOLUTION = (256,256)


input_size = (3, 256, 256)

# Display the model structure


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from generator import Generator, Discriminator

epochs = 100000
checkpoint_dir = 'elves_SA4'
# Hyperparameters
batch_size = 32
nz = 100
# beta1 = 0.5  # For Adam optimizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Create the generator and discriminator
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

summary(generator, input_size=(1,nz))
summary(discriminator, input_size=input_size)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

from torchvision.transforms import functional as TF

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2*x - 1),  # Normalize to [-1, 1]
])

train_dataset = datasets.ImageFolder("/mnt/data/profile", transform=transform)
# Load ImageNet dataset
train_size = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print(f"Number of training images: {train_size}")

def save_images(images, epoch, max_images=20, save_dir="generated_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert the images for visualization
    images = images.detach().cpu().numpy()[:max_images]  # Cap the number of images
    images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    images = np.transpose(images, (0, 2, 3, 1))  # Change from (batch, channel, height, width) to (batch, height, width, channel)

    num_images = len(images)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(4, 5, i + 1)  # This assumes you're showing up to 20 images in a 4x5 grid
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()  # Close the plot to free up resources


real_label = 1.0
fake_label = 0.0

if os.path.exists(f"{checkpoint_dir}/best_generator_model.pth"):
    generator.load_state_dict(torch.load(f"{checkpoint_dir}/best_generator_model.pth"))
else:
    print("Generator weights not found!")

if os.path.exists(f"{checkpoint_dir}/best_discriminator_model.pth"):
    discriminator.load_state_dict(torch.load(f"{checkpoint_dir}/best_discriminator_model.pth"))
else:
    print("Discriminator weights not found!")


# Losses and optimizers
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0, 0.9))
lambda_gp = 10  # Gradient penalty coefficient

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

use_amp = True  # Set this to False if you don't want to use AMP

if use_amp:
    scaler = GradScaler()
else:
    scaler = None  # This will help avoid using the scaler if AMP is turned off


for epoch in range(epochs):
    err_d = 0.0
    err_g = 0.0
    for i,  (data, _) in enumerate(train_loader, 0):
        # --- Update Discriminator/Critic ---
        for _ in range(5):  # Update discriminator 5 times for each generator update
            discriminator.zero_grad()

            real = data.to(device)
            
            # Use autocast for mixed-precision forward pass
            with autocast(enabled=use_amp):
                # Loss from real images
                d_real = discriminator(real)
                err_d_real = -torch.mean(d_real)

                # Loss from fake images
                noise = torch.randn(batch_size, nz, device=device)
                fake = generator(noise)
                d_fake = discriminator(fake.detach())
                err_d_fake = torch.mean(d_fake)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real.data, fake.data)
                err_d = err_d_real + err_d_fake + lambda_gp * gradient_penalty

            # Gradient scaling for backward pass
            if use_amp:
                scaler.scale(err_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
            else:
                err_d.backward()
                optimizer_d.step()

        # --- Update Generator ---
        generator.zero_grad()

        # Use autocast for mixed-precision forward pass
        with autocast(enabled=use_amp):
            noise = torch.randn(batch_size, nz, device=device)
            fake = generator(noise)
            d_fake = discriminator(fake)
            err_g = -torch.mean(d_fake)
        
        # Gradient scaling for backward pass
        if use_amp:
            scaler.scale(err_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
        else:
            err_g.backward()
            optimizer_g.step()

    if epoch % 10 == 0 and epoch > 0:  # Save every 10 epochs, but you can adjust this
        with torch.no_grad():
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)
            save_images(fake_images, epoch, 20, f"{checkpoint_dir}/generated_images")

    # Validation loop
    print(f"\rEpoch {epoch}/{epochs} | Validation D Loss: {err_d:.4f} | Validation G Loss: {err_g.item():.4f}", end = '')
    # validate(batch_size, epochs, device, generator, discriminator, criterion, val_loader, real_label, fake_label, epoch)
    torch.save(discriminator.state_dict(), f"{checkpoint_dir}/best_discriminator_model.pth")
    torch.save(generator.state_dict(), f"{checkpoint_dir}/best_generator_model.pth")




