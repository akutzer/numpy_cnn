import numpy as np


class Adam():
    def __init__(self, parameters, learning_rate, betas=(.9, .999), eps=1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.betas = betas
        self.eps = eps

        self.cache = self._init_cache()
        self.t = 1

    def _init_cache(self):
        return {i: {"m":[np.zeros(parameter.weight.shape),
                         np.zeros(parameter.bias.shape)],
                    "v":[np.zeros(parameter.weight.shape),
                         np.zeros(parameter.bias.shape)]}
                    for i, parameter in enumerate(self.parameters)}

    def update(self):
        for i, parameter in enumerate(self.parameters):
            # calculate weight update
            m = self.betas[0]*self.cache[i]["m"][0] \
                + (1-self.betas[0])*parameter.w_grad
            mt = m / (1-self.betas[0]**self.t)

            v = self.betas[1]*self.cache[i]["v"][0]\
                + (1-self.betas[1])*parameter.w_grad**2
            vt = v / (1-self.betas[1]**self.t)

            weight_update = -self.lr * mt / (np.sqrt(vt) + self.eps)
            self.cache[i]["m"][0] = m
            self.cache[i]["v"][0] = v

            # calculate bias update
            m = self.betas[0]*self.cache[i]["m"][1] \
                + (1-self.betas[0])*parameter.b_grad
            mt = m / (1-self.betas[0]**self.t)

            v = self.betas[1]*self.cache[i]["v"][1] \
                + (1-self.betas[1])*parameter.b_grad**2
            vt = v / (1-self.betas[1]**self.t)

            bias_update = -self.lr * mt / (np.sqrt(vt) + self.eps)
            self.cache[i]["m"][1] = m
            self.cache[i]["v"][1] = v

			# apply updates
            parameter.weight += weight_update
            parameter.bias += bias_update
            self.t += 1

            parameter._reset_gradients()
