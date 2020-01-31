import numpy as np


class Adagrad():
    def __init__(self, parameters, learning_rate, eps=1e-10):
        self.parameters = parameters
        self.lr = learning_rate
        self.eps = eps

        self.cache = self._init_cache()
        self.t = 1

    def _init_cache(self):
        return {i: [np.zeros(parameter.weight.shape),
                    np.zeros(parameter.bias.shape)]
                    for i, parameter in enumerate(self.parameters)}

    def update(self):
        for i, parameter in enumerate(self.parameters):
            # calculate weight update
            self.cache[i][0] += parameter.w_grad**2
            weight_update = -self.lr * parameter.w_grad \
                            / (np.sqrt(self.cache[i][0]) + self.eps)

            # calculate bias update
            self.cache[i][1] += parameter.b_grad**2
            bias_update = -self.lr * parameter.b_grad \
                          / (np.sqrt(self.cache[i][1]) + self.eps)

			# apply updates
            parameter.weight += weight_update
            parameter.bias += bias_update

            parameter._reset_gradients()
