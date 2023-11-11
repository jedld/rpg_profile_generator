import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Implements a SAGAN

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Multiplicative factor to scale the attention map
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B: batch size, C: channel, W: width, H: height
        B, C, W, H = x.size()
        
        # Query and Key have size (B, C//8, W*H)
        Q = self.query(x).view(B, -1, W*H).permute(0, 2, 1)
        K = self.key(x).view(B, -1, W*H)
        
        # Attention map
        attn = self.softmax(torch.bmm(Q, K))
        
        # Value has size (B, C, W*H)
        V = self.value(x).view(B, -1, W*H)
        
        # Weighted value
        y = torch.bmm(V, attn.permute(0, 2, 1))
        
        # Reshape back to (B, C, W, H)
        y = y.view(B, C, W, H)
        
        # Scale and add the residual connection
        out = self.gamma * y + x
        
        return out

class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(nz, 512*4*4)

        # Initial upsampling
        self.init_upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 64, 1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # 4x4 -> 8x8
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # 8x8 -> 16x16
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        # 16x16 -> 32x32 with skip connection
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # 32x32 -> 64x64 with skip connection
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(512 + 64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        
        x1 = self.init_upsample(x)
        x2 = self.upsample1(x1)
        x3 = self.upsample2(x2)

        # Adding the skip connection (concatenating feature maps)
        x3_with_skip = torch.cat([x3, x2], dim=1)
        x4 = self.upsample3(x3_with_skip)

        x4_with_skip = torch.cat([x4, x1], dim=1)
        out = self.upsample4(x4_with_skip)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # SelfAttention(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1)  # Average over the spatial dimensions
        )

    def forward(self, x):
            x = self.main(x)
            return x.view(x.size(0), -1)  # Flatten to [batch_size, 1]
