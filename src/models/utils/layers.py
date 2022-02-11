import torch
import torch.nn as nn


def make_activation_layer(activation_type='relu'):
    act_type = activation_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    if act_type == 'leaky':
        return nn.LeakyReLU()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'tanh':
        return nn.Tanh()

    assert False, 'Invalid Layer'


def make_layer(desc_line, prev_size):
    desc = desc_line.strip().split()
    layer_name = desc[0].lower()

    if layer_name == 'conv':
        kernel_count = int(desc[1])
        kernel_size = int(desc[2])
        stride = int(desc[3])
        padding = int(desc[4])
        return nn.Conv2d(prev_size, kernel_count, kernel_size, stride, padding), kernel_count

    elif layer_name.startswith('global'):
        if 'max' in layer_name:
            return nn.AdaptiveMaxPool2d(1), prev_size
        elif 'avg' in layer_name:
            return nn.AdaptiveAvgPool2d(1), prev_size

    elif layer_name.endswith('pool'):
        pool_size = int(desc[1])
        stride = int(desc[2])
        padding = int(desc[3])
        if 'max' in layer_name:
            return nn.MaxPool2d(pool_size, stride, padding), prev_size
        elif 'avg' in layer_name:
            return nn.AvgPool2d(pool_size, stride, padding), prev_size

    elif layer_name == 'flatten':
        return nn.Flatten(), prev_size
    elif layer_name == 'softmax':
        return nn.Softmax(dim=1), prev_size
    return make_activation_layer(layer_name), prev_size


if __name__ == '__main__':
    x = torch.ones(32, 3, 32, 32)
    conv = nn.Conv2d(3, 6, 5, 1, 0)
    y = conv(x)
    print(y.shape)
    print(conv.weight.shape)
    print(conv.bias.shape)

    glbMaxPool = nn.AdaptiveAvgPool2d(1)
    y = glbMaxPool(y)
    print(y.shape)

    flatten = nn.Flatten()
    y = flatten(y)
    print(y.shape)

    soft = nn.Softmax(dim=1)
    y = soft(y)
    print(y.shape)
    print(y)
