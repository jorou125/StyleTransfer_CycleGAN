import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_R_blocks=6):
        super().__init__()
        layers = []
        c7s1_64 = self.build_c7s1_k(in_dim=3, out_dim=64)
        d128 = self.build_dk(in_dim=64, out_dim=128)
        d256 = self.build_dk(in_dim=128, out_dim=256)
        layers += [c7s1_64, d128, d256]
        R_blocks = self.build_Rk(dim=256, n_blocks=num_R_blocks)
        layers += R_blocks

        u128 = self.build_uk(in_dim=256, out_dim=128)
        u64 = self.build_uk(in_dim=128, out_dim=64)
        c7s1_3 = self.build_c7s1_k(in_dim=64, out_dim=3) # In github there is no InstanceNorm or ReLU after this layer, why???
        layers += [u128, u64, c7s1_3]
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = self.model(x) 
        return x # Supposedly, there is a torch.tanh here according to implementation, but not mentioned in article...

    def build_c7s1_k(self, in_dim, out_dim):
        c7s1_k = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=7, stride=1, bias=True),
            nn.InstanceNorm2d(out_dim, affine=False),
            nn.ReLU(inplace=True)
        )
        return c7s1_k

    def build_dk(self, in_dim, out_dim):
        dk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, bias=True), 
            nn.InstanceNorm2d(out_dim, affine=False),
            nn.ReLU(inplace=True)
        )
        return dk

    def build_uk(self, in_dim, out_dim):
        uk = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(out_dim, affine=False),
            nn.ReLU(inplace=True)
        )
        return uk

    def build_Rk(self, dim, n_blocks=6):
        R_blocks = []
        for _ in range(n_blocks):
            R_blocks.append(ResidualBlock(dim))
        return R_blocks
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Rk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, bias=True),
            nn.InstanceNorm2d(dim, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, bias=True),
            nn.InstanceNorm2d(dim, affine=False),
        )

    def forward(self, x):
        return x + self.Rk(x)
    
if __name__ == "__main__":
    print("Initializing generator")
    G = Generator(num_R_blocks=9)
    print("Testing generator with random input")
    x = torch.randn((1, 3, 256, 256))
    out = G(x)
    print(out.shape)  # Expected output shape: (1, 3, 256, 256)
    # resulting shape does not have the right shape, currently (1, 3, 253, 253), unless out_padding=1 in upsampling?