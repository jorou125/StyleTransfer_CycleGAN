import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_resolution=256):
        super().__init__()
        c7s1_64 = self.build_c7s1_k(in_dim=3, out_dim=64)
        d128 = self.build_dk(in_dim=64, out_dim=128)
        d256 = self.build_dk(in_dim=128, out_dim=256)
        layers_before_R = [c7s1_64, d128, d256]
        self.model_before_R = nn.Sequential(*layers_before_R)

        num_R_blocks = 6 if img_resolution <= 128 else 9
        R_blocks = self.build_Rk(dim=256, n_blocks=num_R_blocks)
        self.R_blocks = R_blocks

        u128 = self.build_uk(in_dim=256, out_dim=128)
        u64 = self.build_uk(in_dim=128, out_dim=64)
        c7s1_3 = self.build_c7s1_k(in_dim=64, out_dim=3) # In github the stride is 0, why???
        layers_after_R = [u128, u64, c7s1_3]
        self.model_after_R = nn.Sequential(*layers_after_R)
        
        self.model = nn.Sequential(
            self.model_before_R,
            self.R_blocks,
            self.model_after_R
        )

    def forward(self, x):
        x = self.model(x) 
        return x # Supposedly, there is a torch.tanh here according to research, but not mentioned in article...

    def build_c7s1_k(self, in_dim, out_dim):
        c7s1_k = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        return c7s1_k

    def build_dk(self, in_dim, out_dim):
        dk = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),  # Is reflect padding used everywhere or only in dk?
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        return dk

    def build_uk(self, in_dim, out_dim):
        uk = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, padding=1), # Corresponds to the « fractional stride conv »
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        return uk

    def build_Rk(self, dim, n_blocks=6):
        R_blocks = []
        for _ in range(n_blocks):
            R_blocks.append(ResidualBlock(dim))
        return nn.Sequential(*R_blocks)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Rk = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True) # In the github repo, there is no ReLU here. Should we add it or not? Seems more logical with it.
        )

    def forward(self, x):
        return x + self.Rk(x)