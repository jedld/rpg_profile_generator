import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Implements a SAGAN

class MultiheadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiheadSelfAttention, self).__init__()

        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        self.head_dim = in_channels // num_heads
        self.num_heads = num_heads
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        
        # Multi-head split
        Q = self.query(x).view(B, self.num_heads, self.head_dim, W*H).permute(0, 2, 1, 3)
        K = self.key(x).view(B, self.num_heads, self.head_dim, W*H).permute(0, 2, 3, 1)
        V = self.value(x).view(B, self.num_heads, self.head_dim, W*H).permute(0, 2, 1, 3)


        # Scaled dot-product attention
        attn = self.softmax((Q @ K) / (self.head_dim ** 0.5))
        y = attn @ V
        y = y.permute(0, 2, 1, 3).contiguous().view(B, C, W, H)
        
        # Project back to the original size and add the residual connection
        out = self.out(self.gamma * y) + x
        return out
    
class Generator(nn.Module):
    def __init__(self, nz=100):  # nz is the size of the latent vector (noise)
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(nz, 512*4*4)  # Adjusted for 4x4 feature maps
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            # MultiheadSelfAttention(256),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(1024, 2048, 4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),

 
            # 32x32 -> 64x64
            nn.ConvTranspose2d(2048, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # reshape to 4x4 feature maps
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 2048, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2048, 1024, 4, stride=2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # MultiheadSelfAttention(256),

            nn.Conv2d(1024, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)  # Average over the spatial dimensions
        )

    def forward(self, x):
            x = self.main(x)
            return x.view(x.size(0), -1)  # Flatten to [batch_size, 1]
    
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)    
