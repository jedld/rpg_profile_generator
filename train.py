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

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 26
IMAGE_SIZE = 256
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 10000
FEATURES_CRITIC = 32
FEATURES_GEN = 32
CRITIC_ITERATIONS = 5
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

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

if os.path.exists(f"best_generator_model.pth"):
    gen.load_state_dict(torch.load(f"best_generator_model.pth"))
else:
    print("Generator weights not found!")

if os.path.exists(f"best_discriminator_model.pth"):
    critic.load_state_dict(torch.load(f"best_discriminator_model.pth"))
else:
    print("Discriminator weights not found!")

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


gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # Use autocast for mixed-precision forward pass
            with autocast(enabled=use_amp):
                noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)

                        # Gradient scaling for backward pass
            if use_amp:
                scaler.scale(loss_critic).backward()
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

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    # Print losses occasionally and print to tensorboard
    if epoch % 5 == 0 and epoch > 0:
        with torch.no_grad():
            with autocast(enabled=use_amp):
                noise = torch.randn(20, Z_DIM).to(device)
                fake_images = gen(noise)
                save_images(fake_images, epoch, 20, f"generated_images")
    torch.save(critic.state_dict(), f"best_discriminator_model.pth")
    torch.save(gen.state_dict(), f"best_generator_model.pth")            