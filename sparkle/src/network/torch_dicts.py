# Generic imports
import torch
import torch.nn as tnn

torch_activations = {
    'relu': tnn.ReLU(),
    'prelu': tnn.PReLU(),
    'relu6': tnn.ReLU6(),
    'lrelu': tnn.LeakyReLU(),
    'selu': tnn.SELU(),
    'mish': tnn.Mish(),
    'sigmoid': tnn.Sigmoid(),
    'tanh': tnn.Tanh(),
    'softmax': tnn.Softmax(),
    'logsoftmax': tnn.LogSoftmax(),
    'linear': None
}

# Helper function to append a layer to a ModuleList
def add_fc_layer(net, dim_in, dim_out, activation, dropout=0.0):

    n  = 0
    net.append(tnn.Linear(dim_in, dim_out))
    n += 1
    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1
    if (dropout != 0.0):
        net.append(tnn.Dropout(dropout))
        n += 1

    return n

# Helper function to append a layer to a ModuleList
def add_conv2d_layer(net, in_ch, out_ch, kernel, stride, padding, activation, w, h):

    net.append(tnn.Conv2d(in_ch, out_ch, kernel, stride, padding))
    n = 1

    w_out = shape(w, kernel, padding, stride)
    h_out = shape(h, kernel, padding, stride)

    if (activation != "linear"):
        net.append(torch_activations[activation])
        n += 1

    return n, w_out, h_out

# Helper function to append a layer to a ModuleList
def add_maxpool2d_layer(net, kernel, stride, w, h):

    net.append(tnn.MaxPool2d(kernel, stride))
    n = 1

    w_out = shape(w, kernel, 0, stride)
    h_out = shape(h, kernel, 0, stride)

    return n, w_out, h_out

# Compute output shape for conv layers
# w = width (and height, assuming square images)
# k = kernel size
# p = padding
# s = stride
# d = dilation
def shape(w, k=3, p=0, s=1, d=1):

    return int((w + 2*p - d*(k - 1) - 1)/s + 1)
