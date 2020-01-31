from CNN.test_layers import *
import numpy as np
from tqdm import tqdm


def test_modules(itter=5):
    print("\nSmall Errors are just rounding Errors!")

    # test convs
    print("\nConvolution:")
    conv_check = {"out":0, "grad":0, "weight":0, "bias":0}
    for _ in tqdm(range(itter)):
        filter_size, stride = np.random.randint(1, 10, (2,))
        padding = int(np.random.randint(0, 5, (1,)))
        b, c_in, c_out, h, w = np.random.randint(filter_size, 49, (5,))
        input = np.random.randn(b, c_in, h, w)

        out_same, grad_same, weight_same, bias_same = \
                        test_conv(input, c_out, filter_size, stride, padding)
        if (out_same, grad_same, weight_same, bias_same) \
            == (False, False, False, False):
            print("ALL WRONGE!!")
        conv_check["out"] += out_same/itter
        conv_check["grad"] += grad_same/itter
        conv_check["weight"] += weight_same/itter
        conv_check["bias"] += bias_same/itter
    for k, v in conv_check.items():
        print(f"\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")

    # test transposed_convs
    print("\nTransposed Convolution:")
    trans_conv_check = {"out":0, "grad":0, "weight":0, "bias":0}
    for _ in tqdm(range(itter)):
        filter_size, stride = np.random.randint(1, 6, (2,))
        b, c_in, c_out, h, w = np.random.randint(filter_size, 33, (5,))
        max_padding = (stride*(min(h,w)-1)+filter_size+1)//2
        padding = int(np.random.randint(0, min(max_padding, 10), (1,)))
        input = np.random.randn(b, c_in, h, w)

        out_same, grad_same, weight_same, bias_same = \
            test_trans_conv(input, c_out, filter_size, stride, padding)
        if (out_same, grad_same, weight_same, bias_same) \
            == (False, False, False, False):
            print("ALL WRONGE!!")
        trans_conv_check["out"] += out_same/itter
        trans_conv_check["grad"] += grad_same/itter
        trans_conv_check["weight"] += weight_same/itter
        trans_conv_check["bias"] += bias_same/itter
    for k, v in trans_conv_check.items():
        print(f"\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")

    # test pool
    print("\nMaxPooling:")
    pool_check = {"out":0, "grad":0}
    for _ in tqdm(range(itter)):
        filter_size, stride = np.random.randint(1, 10, (2,))
        b, c_in, h, w = np.random.randint(filter_size, 65, (4,))
        input = np.random.randn(b, c_in, h, w)

        out_same, grad_same = test_pool(input, filter_size, stride)
        if (out_same, grad_same) == (False, False):
            print("ALL WRONGE!!")
        pool_check["out"] += out_same/itter
        pool_check["grad"] += grad_same/itter
    for k, v in pool_check.items():
        print(f"\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")

    # test linear
    print("\nLinear:")
    linear_check = {"out":0, "grad":0, "weight":0, "bias":0}
    for _ in tqdm(range(itter)):
        b, feat_in, feat_out = np.random.randint(100, 1000, (3,))
        input = np.random.randn(b, feat_in)

        out_same, grad_same, weight_same, bias_same = \
            test_linear(input, feat_in, feat_out)
        if (out_same, grad_same, weight_same, bias_same) \
            == (False, False, False, False):
            print("ALL WRONGE!!")
        linear_check["out"] += out_same/itter
        linear_check["grad"] += grad_same/itter
        linear_check["weight"] += weight_same/itter
        linear_check["bias"] += bias_same/itter
    for k, v in linear_check.items():
        print(f"\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")

    # test activation
    print("\nActivations:")
    activations = ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "Softmax", "LogSoftmax"]
    for activation in activations:
        activation_check = {"out":0, "grad":0}
        for _ in range(itter):
            b, feat_in, feat_out = np.random.randint(100, 1000, (3,))
            input = np.random.randn(b, feat_in)

            out_same, grad_same = test_activation(input, activation)
            if (out_same, grad_same) == (False, False):
                print("ALL WRONGE!!")
            activation_check["out"] += out_same/itter
            activation_check["grad"] += grad_same/itter

        print(f"\t{activation}")
        for k, v in activation_check.items():
            print(f"\t\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")

    #test loss
    print("\nLosses:")
    losses =  ["MSELoss", "MAELoss", "BCELoss", "CrossEntropyLoss", "SoftmaxLoss"]
    for loss in losses:
        loss_check = {"out":0, "grad":0}
        for i in range(itter):
            if loss in ["MSELoss", "MAELoss"]:
                if i%2 == 0:
                    b, feat = np.random.randint(100, 1000, (2,))
                    x, y = np.random.randn(b, feat), np.random.randn(b, feat)
                if i%2 + 1 == 0:
                    b, d, h, w = np.random.randint(10, 100, (2,))
                    x, y = np.random.randn(b, d, h, w), np.random.randn(b, d, h, w)
                out_same, grad_same = test_loss(x, y, loss)

            elif loss == "BCELoss":
                b = np.random.randint(1, 1000)
                x, y = np.random.rand(b, 1), np.random.rand(b, 1)
                out_same, grad_same = test_loss(x, y, loss)

            elif loss == "CrossEntropyLoss":
                b, feat = np.random.randint(100, 1000, (2,))
                x, y = np.random.rand(b, feat), np.random.randint(0, feat, b)
                out_same, grad_same = test_loss(x, y, loss)

            elif loss == "SoftmaxLoss":
                b, feat = np.random.randint(100, 1000, (2,))
                x, y = np.random.randn(b, feat), np.random.randint(0, feat, b)
                out_same, grad_same = test_softmax_loss(x, y)

            if (out_same, grad_same) == (False, False):
                print("ALL WRONGE!!")
            loss_check["out"] += out_same/itter
            loss_check["grad"] += grad_same/itter
        print(f"\t{loss}")
        for k, v in loss_check.items():
            print(f"\t\t{k}: {round(v*itter)}/{itter}\t{round(v*100, 3)}%")


test_modules(itter=150)
