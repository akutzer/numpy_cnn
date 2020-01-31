import numpy as np


class RMSprop():
    def __init__(self, parameters, learning_rate, decay_rate=.99, eps=1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.dr = decay_rate
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
            self.cache[i][0] = self.dr*self.cache[i][0] \
                               + (1-self.dr) * parameter.w_grad**2
            weight_update = -self.lr * parameter.w_grad \
                            / (np.sqrt(self.cache[i][0]) + self.eps)

            # calculate bias update
            self.cache[i][1] = self.dr*self.cache[i][1] \
                               + (1-self.dr) * parameter.b_grad**2
            bias_update = -self.lr * parameter.b_grad \
                          / (np.sqrt(self.cache[i][1]) + self.eps)

			# apply updates
            parameter.weight += weight_update
            parameter.bias += bias_update

            parameter._reset_gradients()
