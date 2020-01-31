import torch
import torch.nn as nn

from .Layer.Conv import Conv
from .Layer.TransposedConv import TransposedConv
from .Layer.MaxPool import MaxPool
from .Layer.Linear import Linear
from .Layer.Activation import *
from .Layer.Loss import *

def forward_backward(module1, module2, x):
    t_x = torch.from_numpy(x).requires_grad_()
    out1 = module1(x)
    out2 = module2(t_x)
    if out1.shape != out2.shape:
        raise ValueError()

    grad = np.random.randn(*out1.shape)
    t_grad = torch.from_numpy(grad)
    grad1 = module1.gradient(grad)
    out2.backward(t_grad)
    grad2 = t_x.grad

    out_same = np.array_equal(np.float32(out2.detach().numpy()), np.float32(out1))
    grad_same = np.array_equal(np.float32(grad2.numpy()), np.float32(grad1))

    return out_same, grad_same, module1, module2

def forward_backward_loss(module1, module2, x, y, long=False):
    t_x = torch.from_numpy(x).requires_grad_()
    t_y = torch.from_numpy(y)
    if long:
        t_y = t_y.long()

    out1 = module1(x, y)
    out2 = module2(t_x, t_y)
    if out1.shape != out2.shape:
        raise ValueError()

    grad1 = module1.gradient()
    out2.backward()
    grad2 = t_x.grad

    out_same = np.array_equal(np.float32(out2.detach().numpy()), np.float32(out1))
    grad_same = np.array_equal(np.float32(grad2.numpy()), np.float32(grad1))

    return out_same, grad_same, module1, module2

def test_conv(x, c_out, filter_size, stride, padding):
    module1 = Conv(x.shape[1], c_out, filter_size, stride=stride, padding=padding)
    module2 = nn.Conv2d(x.shape[1], c_out, filter_size, stride=stride, padding=padding)
    module2.weight.data = torch.from_numpy(module1.weight)
    module2.bias.data = torch.from_numpy(module1.bias)

    out_same, grad_same, module1, module2 = forward_backward(module1, module2, x)

    weight_same = np.array_equal(np.float32(module2.weight.grad.numpy()), np.float32(module1.w_grad))
    bias_same = np.array_equal(np.float32(module2.bias.grad.numpy()), np.float32(module1.b_grad))

    return out_same, grad_same, weight_same, bias_same

def test_trans_conv(x, c_out, filter_size, stride, padding):
    module1 = TransposedConv(x.shape[1], c_out, filter_size, stride=stride, padding=padding)
    module2 = nn.ConvTranspose2d(x.shape[1], c_out, filter_size, stride=stride, padding=padding)
    module2.weight.data = torch.from_numpy(module1.weight.transpose(1,0,2,3))
    module2.bias.data = torch.from_numpy(module1.bias)

    out_same, grad_same, module1, module2 = forward_backward(module1, module2, x)

    weight_same = np.array_equal(np.float32(module2.weight.grad.numpy().transpose(1,0,2,3)), np.float32(module1.w_grad))
    bias_same = np.array_equal(np.float32(module2.bias.grad.numpy()), np.float32(module1.b_grad))

    return out_same, grad_same, weight_same, bias_same

def test_pool(x, pool_size, stride):
    pool_size, stride = int(pool_size), int(stride)
    module1 = MaxPool(pool_size, stride)
    module2 = nn.MaxPool2d(pool_size, stride)

    out_same, grad_same, _, _ = forward_backward(module1, module2, x)

    return out_same, grad_same

def test_linear(x, feat_in, feat_out):
    module1 = Linear(feat_in, feat_out)
    module2 = nn.Linear(feat_in, feat_out)
    module2.weight.data = torch.from_numpy(module1.weight.T)
    module2.bias.data = torch.from_numpy(module1.bias)

    out_same, grad_same, module1, module2 = forward_backward(module1, module2, x)

    weight_same = np.array_equal(np.float32(module2.weight.grad.T.numpy()), np.float32(module1.w_grad))
    bias_same = np.array_equal(np.float32(module2.bias.grad.numpy()), np.float32(module1.b_grad))

    return out_same, grad_same, weight_same, bias_same

def test_activation(x, name, axis=1):
    activations = {"ReLU": [ReLU(), nn.ReLU()],
                   "LeakyReLU": [LeakyReLU(), nn.LeakyReLU()],
                   "Tanh": [Tanh(), nn.Tanh()],
                   "Sigmoid": [Sigmoid(), nn.Sigmoid()],
                   "Softmax": [Softmax(axis), nn.Softmax(dim=axis)],
                   "LogSoftmax": [LogSoftmax(axis), nn.LogSoftmax(dim=axis)]}
    module1, module2 = activations[name]
    out_same, grad_same, _, _ = forward_backward(module1, module2, x)

    return out_same, grad_same

def test_loss(x, y, name):
    losses = {"MSELoss": [MSELoss(), nn.MSELoss()],
              "MAELoss": [MAELoss(), nn.L1Loss()],
              "BCELoss": [BCELoss(), nn.BCELoss()],
              "CrossEntropyLoss": [CrossEntropyLoss(), nn.CrossEntropyLoss()]}
    module1, module2 = losses[name]
    if name == "CrossEntropyLoss":
        out_same, grad_same, _, _ = forward_backward_loss(module1, module2, x, y, True)
    else:
        out_same, grad_same, _, _ = forward_backward_loss(module1, module2, x, y)

    return out_same, grad_same

def test_softmax_loss(x, y):
    module1, module2_1, module2_2 = SoftmaxLoss(axis=1), nn.LogSoftmax(dim=1), nn.NLLLoss()

    t_x = torch.from_numpy(x).requires_grad_()
    t_y = torch.from_numpy(y).long()

    out1 = module1(x, y)
    out2 = module2_2(module2_1(t_x), t_y)
    if out1.shape != out2.shape:
        raise ValueError()

    grad1 = module1.gradient()
    out2.backward()
    grad2 = t_x.grad

    out_same = np.array_equal(np.float32(out2.detach().numpy()), np.float32(out1))
    grad_same = np.array_equal(np.float32(grad2.numpy()), np.float32(grad1))

    return out_same, grad_same
