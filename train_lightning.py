import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from utils import gradient_penalty
from pytorch_lightning import Trainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

class WGAN_GP(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.generator = Generator(self.hparams.z_dim)
        self.critic = Discriminator()

        # Enable manual optimization
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real, _ = batch
        real = real.to(self.device)
        real.requires_grad_()  # Enable gradients for real images

        opt_gen, opt_critic = self.optimizers()

        # Train Discriminator (Critic)
        for _ in range(self.hparams.critic_iterations):
            z = torch.randn(real.size(0), self.hparams.z_dim).type_as(real)
            fake = self(z).detach()
            critic_real = self.critic(real).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)
            gp = gradient_penalty(self.critic, real, fake, device=self.device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.hparams.lambda_gp * gp

            opt_critic.zero_grad()
            self.manual_backward(loss_critic)
            opt_critic.step()

        # Train Generator
        z = torch.randn(real.size(0), self.hparams.z_dim).type_as(real)
        fake = self(z)
        critic_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(critic_fake)

        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        opt_gen.step()

        self.log('critic_loss', loss_critic)
        self.log('generator_loss', loss_gen)

    def on_train_epoch_end(self):
        # Generate a batch of images
        z = torch.randn(20, self.hparams.z_dim, device=self.device)
        fake_images = self(z).detach()

        # Save the images
        self.save_images(fake_images, epoch=self.current_epoch)

    def save_images(self, images, epoch, max_images=20, save_dir="generated_images"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        images = images.cpu().numpy()[:max_images]
        images = (images + 1) / 2
        images = np.transpose(images, (0, 2, 3, 1))

        plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(4, 5, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.tight_layout()
        plt.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
        plt.close()

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gen, betas=(0.0, 0.9))
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr_critic, betas=(0.0, 0.9))
        return opt_gen, opt_critic

class WGAN_GPDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

# Hyperparameters
hparams = {
    'z_dim': 100,
    'lr_gen': 1e-4,
    'lr_critic': 1e-4,
    'lambda_gp': 10,
    'critic_iterations': 3
}

IMAGE_SIZE = 64
CHANNELS_IMG = 3

# Data preparation
transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ])  # your transforms here

dataset = datasets.ImageFolder(root="/mnt/data/profile", transform=transforms)
data_module = WGAN_GPDataModule(dataset, batch_size=64)

# Model
model = WGAN_GP(hparams)


checkpoint_path = "checkpoints"
if os.path.exists(checkpoint_path):
    model = WGAN_GP.load_from_checkpoint(checkpoint_path + "/checkpoint.ckpt")


# Define a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path,
    filename="checkpoint"
)

# Trainer
trainer = Trainer(max_epochs=10000, callbacks=[checkpoint_callback], log_every_n_steps=5)
trainer.fit(model, data_module)
