import torch
import torch.nn as nn
import functools

    
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [
            nn.Conv2d(nf, nf, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    print("Generating Discriminator Model")
    D = Discriminator()
    print("Discriminator Model Generated")
    x = torch.randn((5, 3, 256, 256))
    print("Running Discriminator on random input")
    y = D(x)
    print(y.shape)  # Expected shape: (5, 1, 30, 30)
    