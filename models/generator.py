import torch
import torch.nn as nn
import functools

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        layers = []
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            norm_layer(dim)
        ]  
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True), n_blocks=9):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), 
            norm_layer(ngf), 
            nn.ReLU(True)
            ]

        n_du_sample = 2
        for i in range(n_du_sample):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_du_sample
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    norm_layer=norm_layer,
                    use_bias=use_bias
                )
            ]

        for i in range(n_du_sample):
            mult = 2 ** (n_du_sample - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()      
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    print("Generating Generator Model")
    G = Generator()
    print("Generator Model Generated")
    x = torch.randn((5, 3, 256, 256))
    print("Running Generator on random input")
    y = G(x)
    print(y.shape)  # Expected shape: (5, 3, 256, 256)