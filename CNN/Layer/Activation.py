import numpy as np


class ReLU():
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.forward_in = x
        return np.maximum(0, x)

    def gradient(self, prev_grad):
        if self.forward_in.shape != prev_grad.shape:
            prev_grad = prev_grad.reshape(self.forward_in.shape)
        prev_grad[self.forward_in<0]=0
        return prev_grad


class LeakyReLU():
    def __init__(self, alpha=.01):
        self.alpha = alpha

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.forward_in = x
        return np.where((x>0)==True, x, x*self.alpha)

    def gradient(self, prev_grad):
        if self.forward_in.shape != prev_grad.shape:
            prev_grad = prev_grad.reshape(self.forward_in.shape)
        return np.where((self.forward_in>0)==True, 1, self.alpha)*prev_grad


class Tanh():
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.forward_in = x
        return np.tanh(x)

    def gradient(self, prev_grad):
        if self.forward_in.shape != prev_grad.shape:
            prev_grad = prev_grad.reshape(self.forward_in.shape)
        return (1-(np.tanh(self.forward_in)**2))*prev_grad


class Sigmoid():
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def sigmoid(self, x):
        return np.where(x>0, 1 / (1+np.exp(-x)), np.exp(x) / (1+np.exp(x)))

    def forward(self, x):
        self.forward_in = x
        return self.sigmoid(x)

    def gradient(self, prev_grad):
        if self.forward_in.shape != prev_grad.shape:
            prev_grad = prev_grad.reshape(self.forward_in.shape)
        return self.sigmoid(self.forward_in)*(1-self.sigmoid(self.forward_in))\
               *prev_grad


class Softmax():
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        self.out = exps / np.sum(exps, axis=self.axis, keepdims=True)
        return self.out

    def gradient(self, prev_grad):
        sum = np.sum(prev_grad*self.out, axis=self.axis, keepdims=True)
        return self.out*(prev_grad-sum)


class LogSoftmax():
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        max = np.max(x, axis=self.axis, keepdims=True)
        self.out = (x - max) - np.log(np.sum(np.exp(x - max),
                    axis=self.axis, keepdims=True))
        return self.out

    def gradient(self, prev_grad):
        sum = np.sum(prev_grad, axis=self.axis, keepdims=True)
        return prev_grad - np.exp(self.out)*sum
