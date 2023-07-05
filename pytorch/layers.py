import torch
import torch.nn as nn
from utility import model_save_onnx, getPadding, weight_initialize

class ConvBnAct(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias, 
            padding_mode,
            activation_layer
        ):
        super().__init__()
        self.padding = getPadding(kernel_size, dilation, padding)
        modules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, groups, bias, padding_mode),
            nn.BatchNorm2d(out_channels),
            activation_layer
        ])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = ConvBnAct(
        in_channels=3, 
        out_channels=16, 
        kernel_size=3, 
        stride=1, 
        padding=True, 
        dilation=1, 
        groups=1, 
        bias=False, 
        padding_mode='zeros', 
        activation_layer=nn.LeakyReLU(inplace=True)
    )
    model.apply(weight_initialize)
    model.eval()
    input_shape = (4, 3, 28, 28)
    model_save_onnx(model, input_shape, "conv_layer", True)