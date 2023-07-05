import torch.nn as nn
from layers import ConvBnAct
from utility import model_save_onnx, getPadding, weight_initialize

class ResidualBlock(nn.Module):
    def __init__(            
            self, 
            in_channels,
            width,
            expansion,
            kernel_size, 
            padding, 
            dilation, 
            groups, 
            bias, 
            padding_mode,
            activation_layer,
            apply_bottleneck,
            downsample
        ):
        super().__init__()
        self.expansion = expansion
        self.apply_bottleneck = apply_bottleneck
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        if self.apply_bottleneck:
            module = nn.ModuleList([
                ConvBnAct(in_channels, width, 1, 1, padding, dilation, groups, bias, padding_mode, activation_layer),
                ConvBnAct(width, width, kernel_size, 1, padding, dilation, groups, bias, padding_mode, activation_layer)
            ])
            self.conv = nn.Conv2d(width, width * self.expansion, 1, self.stride, getPadding(1, dilation, padding), dilation, groups, bias, padding_mode)
            self.bn = nn.BatchNorm2d(width * self.expansion)
            self.act = activation_layer
            self.identity_conv = nn.Conv2d(in_channels, width * self.expansion, 1, self.stride, getPadding(1, dilation, padding), dilation, groups, bias, padding_mode)
            self.identity_bn = nn.BatchNorm2d(width * self.expansion)
        else:
            module = nn.ModuleList([
                ConvBnAct(in_channels, width, kernel_size, 1, padding, dilation, groups, bias, padding_mode, activation_layer)
            ])
            self.conv = nn.Conv2d(width, width, kernel_size, self.stride, getPadding(kernel_size, dilation, padding), dilation, groups, bias, padding_mode)
            self.bn = nn.BatchNorm2d(width)
            self.act = activation_layer
            self.identity_conv = nn.Conv2d(in_channels, width, 1, self.stride, getPadding(1, dilation, padding), dilation, groups, bias, padding_mode)
            self.identity_bn = nn.BatchNorm2d(width)
        self.conv_unit = nn.Sequential(*module)

    def forward(self, x):
        out = self.conv_unit(x)
        out = self.conv(out)
        out = self.bn(out)
        x = self.identity_conv(x)
        x = self.identity_bn(x)
        out = out + x
        out = self.act(out)
        return out
    

if __name__ == "__main__":
    model = ResidualBlock(
        in_channels=3, 
        width=32,
        expansion=4,
        kernel_size=3, 
        padding=True, 
        dilation=1, 
        groups=1, 
        bias=False, 
        padding_mode='zeros', 
        activation_layer=nn.LeakyReLU(inplace=True),
        apply_bottleneck=True,
        downsample=True
    )
    model.apply(weight_initialize)
    model.eval()
    input_shape = (4, 3, 256, 256)
    model_save_onnx(model, input_shape, "residual_block", True)