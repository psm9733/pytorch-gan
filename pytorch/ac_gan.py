import torch
import torch.nn as nn
from blocks import ResidualBlock
from layers import ConvBnAct, LinearBnAct
import pytorch_lightning as pl
from utility import model_save_onnx, getPadding, weight_initialize

class Discriminator(nn.Module):
    def __init__(            
            self,
            in_channels=3,
            class_num=10
        ):
        super().__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        self.stem = nn.Sequential(
            ConvBnAct(in_channels, 32, 3, 2, padding=True, dilation=1, groups=1, bias=True, padding_mode='zeros', activation_layer=nn.ReLU(inplace=True)),
            ConvBnAct(32, 64, 3, 2, padding=True, dilation=1, groups=1, bias=True, padding_mode='zeros', activation_layer=nn.ReLU(inplace=True)),
        )
        self.block1 = nn.Sequential(
            ResidualBlock(        
                in_channels=64, 
                width=64,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=False,
                downsample=True
            ),
        )
        self.block2 = nn.Sequential(
            ResidualBlock(        
                in_channels=64, 
                width=64,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=True
            ),
            ResidualBlock(        
                in_channels=256, 
                width=64,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=False
            )
        )
        self.block3 = nn.Sequential(
            ResidualBlock(        
                in_channels=256, 
                width=64,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=False
            ),
            ResidualBlock(        
                in_channels=256, 
                width=128,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=False
            )
        )

        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self.discriminator = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512, class_num + 1, 3, 1, getPadding(3, 1, 1), 1, 1, True, 'zeros'),
            nn.BatchNorm2d(class_num + 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.discriminator(out)
        return out

class Generator(nn.Module):
    def __init__(            
            self,
            latent_dim,
            latent_out_dim,
            init_resolution=7,
            out_channels=3,
            class_num=1000
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_out_dim = latent_out_dim
        self.init_resolution=init_resolution
        self.out_channels=out_channels
        self.class_num = class_num
        self.mapping_net = nn.ModuleList([
            LinearBnAct(latent_dim, latent_dim*2, nn.ReLU(inplace=True)),
            LinearBnAct(latent_dim*2, latent_dim*4, nn.ReLU(inplace=True)),
            LinearBnAct(latent_dim*4, latent_dim*8, nn.ReLU(inplace=True)),
            LinearBnAct(latent_dim*8, int(latent_out_dim * out_channels * init_resolution * init_resolution), nn.ReLU(inplace=True)),
        ])
        self.block1 = nn.ModuleList([
            ResidualBlock(        
                in_channels=latent_out_dim, 
                width=64,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=False,
                downsample=False
            ),
        ])
        self.block2 = nn.ModuleList([
            ResidualBlock(        
                in_channels=64, 
                width=64,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=False,
                downsample=False
            ),
        ])
        self.block3 = nn.ModuleList([
            ResidualBlock(        
                in_channels=64, 
                width=64,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=False
            ),
            ResidualBlock(        
                in_channels=256, 
                width=64,
                expansion=4,
                kernel_size=3, 
                padding=True, 
                dilation=1, 
                groups=1, 
                bias=False, 
                padding_mode='zeros', 
                activation_layer=nn.ReLU(inplace=True),
                apply_bottleneck=True,
                downsample=False
            )
        ])

        self.mapping_net = nn.Sequential(*self.mapping_net)

        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self.generator = nn.Sequential(
            self.block1,
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.block2,
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.block3
        )
        self.img_gen_head = nn.Sequential(
            nn.Conv2d(256, out_channels, 3, 1, getPadding(3, 1, 1), 1, 1, True, 'zeros'),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.class_head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, self.class_num, 1, 1, getPadding(1, 1, 1), 1, 1, True, 'zeros'),
            nn.BatchNorm2d(self.class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.mapping_net(x)
        z = z.view(z.size()[0], self.latent_out_dim, self.init_resolution, self.init_resolution)
        f = self.generator(z)
        img_out = self.img_gen_head(f)
        cls_out = self.class_head(f)
        return img_out, cls_out
    
class ACGAN(pl.LightningModule):
    def __init__(self, generator, discrimator, lr=1e-3):
        super().__init__()
        self.generator = generator
        self.discrimator = discrimator
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = 0.0
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    latent_dim = 64
    discriminator = Discriminator(
        in_channels=1,
        class_num=10
    )
    discriminator.apply(weight_initialize)
    discriminator.eval()
    input_shape = (4, 1, 28, 28)
    model_save_onnx(discriminator, input_shape, "discriminator", True)

    generator = Generator(
        latent_dim=latent_dim,
        latent_out_dim=32,
        init_resolution=7,
        out_channels=1,
        class_num=10
    )
    generator.apply(weight_initialize)
    generator.eval()
    input_shape = (4, latent_dim)
    model_save_onnx(generator, input_shape, "generator", True)