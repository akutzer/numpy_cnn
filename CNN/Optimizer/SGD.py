import numpy as np


class SGD():
    def __init__(self, parameters, learning_rate, momentum=0,
                 nesterov=False, weight_decay=0):
        self.parameters = parameters
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        self.velocity = self._init_velocity()

    def _init_velocity(self):
        return {i: [np.zeros(parameter.weight.shape),
                    np.zeros(parameter.bias.shape)]
                    for i, parameter in enumerate(self.parameters)}

    def update(self):
        for i, parameter in enumerate(self.parameters):

			# calculate weight and bias updates with nesterov accelerated gradient
            if self.nesterov == True:
				# save for later
                prev_velocity = self.velocity[i]

				# weight update
                self.velocity[i][0] = self.momentum*self.velocity[i][0] \
                                      - self.lr*parameter.w_grad
                weight_update = -self.momentum*prev_velocity[0] \
                                + (1+self.momentum)*self.velocity[i][0]

				# bias update
                self.velocity[i][1] = self.momentum*self.velocity[i][1] \
                                      - self.lr*parameter.b_grad
                bias_update = -self.momentum*prev_velocity[1] \
                              + (1+self.momentum)*self.velocity[i][1]

			# calculate weight and bias updates with momenumt
            else:
				# weight update
                weight_update = self.momentum*self.velocity[i][0]\
                                -self.lr*parameter.w_grad
                self.velocity[i][0] = weight_update

				# bias update
                bias_update = self.momentum*self.velocity[i][1]\
                              -self.lr*parameter.b_grad
                self.velocity[i][1] = bias_update

			# apply updates
            parameter.weight += weight_update
            parameter.bias += bias_update

            parameter._reset_gradients()
