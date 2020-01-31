import numpy as np


class MaxPool():
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x): # x SHAPE : (B, C, H, W)
        if len(x.shape) != 4:
            raise ValueError(
                f"Pooling only supports 4D-Inputs, not {len(x.shape)}D-Inputs")

        # pool SHAPE : (B, C, H_out, W_out)
        pool = np.zeros((*x.shape[:2],
                         int((x.shape[2] - self.pool_size)//self.stride +1),
                         int((x.shape[3] - self.pool_size)//self.stride +1)))
        # mask SHAPE : (B, C, H_out*Pool_Size, W_out*Pool_Size)
        self.mask = np.zeros((*x.shape[:2],
                              pool.shape[2]*self.pool_size,
                              pool.shape[3]*self.pool_size))

        self.shape_in = x.shape
        self.shape_out = pool.shape

        for h in range(0, pool.shape[2]):
            for w in range(0, pool.shape[3]):
                kernel = x[:, :, h*self.stride:h*self.stride+self.pool_size,
                           w*self.stride:w*self.stride+self.pool_size]
                kernel_max = np.amax(kernel, axis=(2,3)) # SHAPE : (B, C)
                pool[:, :, h, w] = kernel_max

                kernel_max = kernel_max[:,:,None,None] # SHAPE : (B, C, 1, 1)
                kernel_max = np.repeat(kernel_max, self.pool_size, axis=3) # SHAPE : (B, C, 1, Pool_Size)
                kernel_max = np.repeat(kernel_max, self.pool_size, axis=2) # SHAPE : (B, C, Pool_Size, Pool_Size)
                kernel_mask = np.equal(kernel, kernel_max)*1
                self.mask[:, :,
                          h*self.pool_size:self.pool_size*(h+1),
                          w*self.pool_size:self.pool_size*(w+1)]\
                          = kernel_mask

        return pool

    def gradient(self, prev_grad):
        # if gradient comes from linear layer
        if prev_grad.shape != self.shape_out:
            # new prev_grad SHAPE : (B, C, H_out, W_out)
            prev_grad = prev_grad.reshape(self.shape_out)

        # grad SHAPE : (B, C, H_in, W_in)
        grad = np.zeros(self.shape_in)

        for h in range(0, prev_grad.shape[2]):
            for w in range(0, prev_grad.shape[3]):
                kernel_grad = prev_grad[:, :, h, w][:, :, None, None]
                kernel_grad = np.repeat(kernel_grad, self.pool_size, axis=3)
                kernel_grad = np.repeat(kernel_grad, self.pool_size, axis=2)
                kernel_mask = self.mask[:, :,
                                        h*self.pool_size:self.pool_size*(h+1),
                                        w*self.pool_size:self.pool_size*(w+1)]
                grad[:, :,
                     h*self.stride:h*self.stride+self.pool_size,
                     w*self.stride:w*self.stride+self.pool_size]\
                     += kernel_mask*kernel_grad

        return grad
