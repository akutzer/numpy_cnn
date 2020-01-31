import numpy as np


class MSELoss():
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y): # x -> prediction | y -> target
        self.x, self.y = x, y
        loss = (y - x)**2
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return np.sum(loss)
        elif self.reduction == "mean":
            return 1/np.prod(x.shape) * np.sum(loss)

    def gradient(self, prev_grad=None):
        grad = 2*(self.x - self.y)
        if self.reduction == "none":
            return grad*prev_grad
        elif self.reduction == "sum":
            return grad
        elif self.reduction == "mean":
            return 1/np.prod(self.x.shape) * grad


class MAELoss():
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y): # x -> prediction | y -> target
        self.x, self.y = x, y
        loss = np.abs(y - x)
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return np.sum(loss)
        elif self.reduction == "mean":
            return 1/np.prod(x.shape) * np.sum(loss)

    def gradient(self, prev_grad=None):
        grad = (self.x - self.y)/np.abs(self.x - self.y)
        if self.reduction == "none":
            return grad*prev_grad
        elif self.reduction == "sum":
            return grad
        elif self.reduction == "mean":
            return 1/np.prod(self.x.shape) * grad


class BCELoss():
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y): # x -> prediction | y -> target
        self.x, self.y = x, y
        loss = -(y*np.log(x) + (1-y)*np.log(1-x))
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return np.sum(loss)
        elif self.reduction == "mean":
            return 1/np.prod(x.shape) * np.sum(loss)

    def gradient(self, prev_grad=None):
        grad = -(self.y/self.x - (1-self.y)/(1-self.x))
        if self.reduction == "none":
            return grad*prev_grad
        elif self.reduction == "sum":
            return grad
        elif self.reduction == "mean":
            return 1/np.prod(self.x.shape) * grad


class CrossEntropyLoss():
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input, target):
        return self.forward(input, target)

    def _make_hot_encoding(self, target, shape):
        zeros = np.zeros(shape)
        for i, t in enumerate(target):
            zeros[i,t] = 1
        return zeros

    def forward(self, x, y): # x -> prediction | y -> target
        self.x = x
        self.hot = self._make_hot_encoding(y, x.shape)
        correct_input = np.sum(self.hot*x, axis=1)
        loss = -correct_input + np.log(np.sum(np.exp(x), axis=1))

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return np.sum(loss)
        elif self.reduction == "mean":
            return 1/x.shape[0] * np.sum(loss)

    def gradient(self, prev_grad=None):
        sum = np.repeat(np.sum(np.exp(self.x), axis=1, keepdims=True),
                        self.x.shape[-1], axis=1)
        grad = -self.hot + np.exp(self.x)/sum

        if self.reduction == "none":
            return grad*prev_grad
        elif self.reduction == "sum":
            return grad
        elif self.reduction == "mean":
            return 1/self.x.shape[0] * grad


class SoftmaxLoss(): # combination of Softmax and CrossEntropyLoss
    def __init__(self, axis, reduction="mean"):
        from .Activation import LogSoftmax
        self.reduction = reduction
        self.LogSoftmax = LogSoftmax(axis)

    def __call__(self, x, y):
        return self.forward(x, y)

    def _make_hot_encoding(self, target, shape):
        zeros = np.zeros(shape)
        for i, t in enumerate(target):
            zeros[i,t] = 1
        return zeros

    def forward(self, x, y): # x -> prediction | y -> target
        self.x = x
        log_softmax = self.LogSoftmax(x)
        self.hot = self._make_hot_encoding(y, x.shape)
        loss = -np.sum(self.hot*log_softmax, axis=1)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return np.sum(loss)
        elif self.reduction == "mean":
            return 1/x.shape[0] * np.sum(loss)

    def gradient(self, prev_grad=None):
        grad = self.LogSoftmax.gradient(-self.hot)
        if self.reduction == "none":
            return grad*prev_grad
        elif self.reduction == "sum":
            return grad
        elif self.reduction == "mean":
            return 1/self.x.shape[0] * grad
