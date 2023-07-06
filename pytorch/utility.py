import torch
import torch._C as _C
import torch.nn as nn
from torchinfo import summary
TrainingMode = _C._onnx.TrainingMode

def model_save_onnx(model, input_shape, name, verbose=True):
    dummy_input = torch.randn(size=input_shape).clone().detach()
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    if verbose:
        summary(model, input_shape)
    print("=================== Saving {} model ===================".format(name))
    torch.onnx.export(model, dummy_input, name + ".onnx", training=TrainingMode.TRAINING, opset_version=11)

def getPadding(kernel_size, dilation, padding):
    if padding:
        return (kernel_size - 1) // 2 * dilation
    else:
        return 0
    
def weight_initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)