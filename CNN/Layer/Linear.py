import numpy as np


class Linear():
    def __init__(self, in_feat, out_feat):
        weight_limit = 1/(in_feat**(1/2))
        self.weight = np.random.rand(in_feat, out_feat)\
                      * 2*weight_limit - weight_limit
        #self.bias = np.zeros(out_feat)
        self.bias = np.random.rand(out_feat)\
                      * 2*weight_limit - weight_limit

        self._reset_gradients()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # if x-SHAPE : (B, C, H, W)
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1)

	    # x SHAPE : (B, Feat_in)
        self.forward_in = x
        out = np.dot(x, self.weight) + self.bias

        return out  # return SHAPE : (B, Feat_out)

    def gradient(self, prev_grad):
        # if prev_grad SHAPE : (B, C, H, W)
        if len(prev_grad.shape) == 4:
            prev_grad = prev_grad.reshape(prev_grad.shape[0], -1)
        # prev_grad SHAPE : (B, Feat_out)

        # calculate gradients for weights and biases
        self.w_grad += np.dot(self.forward_in.T, prev_grad)
        self.b_grad += np.sum(prev_grad, axis=0)

	    # calculate gradient of the input
        grad = np.dot(prev_grad, self.weight.T)

        return grad # return SHAPE : (B, Feat_in)

    def _reset_gradients(self):
        self.w_grad = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)
