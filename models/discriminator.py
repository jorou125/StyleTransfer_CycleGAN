import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim = 3, layer_dims = [64, 128, 256, 512]):
        super().__init__()
        layers = []
        for i, dim in enumerate(layer_dims):
            if dim == layer_dims[0]:
                layers.append(self.build_layer(
                    in_dim=input_dim, out_dim=dim, 
                    kernel_size=4, stride=2, padding=1, 
                    InstanceNorm=False))
            elif dim == layer_dims[-1]: # Last layer has a stride of 1!
                layers.append(self.build_layer(
                    in_dim=layer_dims[i-1], out_dim=dim, 
                    kernel_size=4, stride=1, padding=1,
                    InstanceNorm=True))
            else:
                layers.append(self.build_layer(
                    in_dim=layer_dims[i-1], out_dim=dim, 
                    kernel_size=4, stride=2, padding=1,
                    InstanceNorm=True))
        layers.append(nn.Conv2d(in_channels=layer_dims[-1], out_channels=1, 
                                kernel_size=4, stride=1, padding=1)) # « Produce 1-dimensional output »
        self.model = nn.Sequential(*layers)


    def build_layer(self, in_dim, out_dim, kernel_size, stride, padding=1, InstanceNorm=True):
        layer = [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)]
        if InstanceNorm:
            layer.append(nn.InstanceNorm2d(out_dim))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        return self.model(x) # To revisit, unsure if I need to add torch.sigmoid here (@Rohan, do you know?)
    
if __name__ == "__main__":
    print("Generating Discriminator Model")
    D = Discriminator()
    print("Discriminator Model Generated")
    x = torch.randn((1, 3, 256, 256))
    print("Running Discriminator on random input")
    y = D(x)
    print(y.shape)  # Expected shape: (1, 1, 30, 30)
        