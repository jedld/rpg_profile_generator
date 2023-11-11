"""
Training of WGAN-GP

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
from torch.cuda.amp import autocast, GradScaler
from torchsummary import summary

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
CRITIC_LEARNING_RATE = 1e-4
GENERATOR_LEARNING_RATE = 1e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 10000
FEATURES_CRITIC = 32
FEATURES_GEN = 32
CRITIC_ITERATIONS = 3
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist above and uncomment below for training on CelebA
dataset = datasets.ImageFolder(root="/mnt/data/profile", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM).to(device)
critic = Discriminator().to(device)
initialize_weights(gen)
initialize_weights(critic)

summary(gen, input_size=(1, Z_DIM))
summary(critic, input_size=(3, 64, 64))

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE, betas=(0.0, 0.9))

if os.path.exists(f"best_generator_model.pth"):
    gen.load_state_dict(torch.load(f"best_generator_model.pth"))
    if os.path.exists(f"generator_opt.pth"):
        opt_gen.load_state_dict(torch.load(f"generator_opt.pth"))
else:
    print("Generator weights not found!")

if os.path.exists(f"best_discriminator_model.pth"):
    critic.load_state_dict(torch.load(f"best_discriminator_model.pth"))
    if os.path.exists(f"critic_opt.pth"):
        opt_critic.load_state_dict(torch.load(f"critic_opt.pth"))
else:
    print("Discriminator weights not found!")

start_epoch = 0

if os.path.exists(f"meta.pth"):
    meta = torch.load(f"meta.pth")
    start_epoch = meta["epoch"]

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


# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

use_amp = False  # Set this to False if you don't want to use AMP

if use_amp:
    scaler = GradScaler()
else:
    scaler = None  # This will help avoid using the scaler if AMP is turned off


for epoch in range(start_epoch, NUM_EPOCHS):
    gen.train()
    critic.train()

    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # Use autocast for mixed-precision forward pass
            with autocast(enabled=use_amp):
                critic.zero_grad()
                noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                fake = gen(noise)
                
                # Apply the same augmentations to the fake images
                if random.random() < 0.5:  # 50% chance of flipping
                    fake = torch.flip(fake, [3])  # Assuming [B, C, H, W] format

                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )

            if use_amp:
                scaler.scale(loss_critic).backward(retain_graph=True)
                scaler.step(opt_critic)
                scaler.update()
            else:
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen.zero_grad()
        with autocast(enabled=use_amp):
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

        if use_amp:
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
        else:
            loss_gen.backward()
            opt_gen.step()
                
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            step += 1
    # Print losses occasionally and print to tensorboard
    if epoch % 10 == 0 and epoch > 0:
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )
        gen.eval()
        with torch.no_grad():
            noise = torch.randn(20, Z_DIM).to(device)
            fake_images = gen(noise)
            save_images(fake_images, epoch, 20, f"generated_images")
    torch.save(critic.state_dict(), f"best_discriminator_model.pth")
    torch.save(gen.state_dict(), f"best_generator_model.pth") 
    torch.save(opt_gen.state_dict(), f"generator_opt.pth")
    torch.save(opt_critic.state_dict(), f"critic_opt.pth")
    torch.save({ "epoch": epoch }, f"meta.pth")