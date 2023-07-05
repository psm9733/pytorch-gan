import torch
import torch._C as _C
from torchinfo import summary
TrainingMode = _C._onnx.TrainingMode

def model_save_onnx(model, input_shape, name, verbose=True):
    dummy_input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).clone().detach()
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    if verbose:
        summary(model, input_shape)
    print("=================== Saving {} model ===================".format(name))
    torch.onnx.export(model, dummy_input, name + ".onnx", training=TrainingMode.TRAINING)

def getPadding(kernel_size, dilation, padding):
    if padding:
        return (kernel_size - 1) // 2 * dilation
    else:
        return 0