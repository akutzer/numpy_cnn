import numpy as np


class Conv():
    def __init__(self, in_chan, out_chan, filter_size, stride=1, padding=0):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.out_chan = out_chan

        weight_limit = 1 / ((in_chan*filter_size*filter_size)**(1/2))
        self.weight = np.random.rand(out_chan, in_chan, filter_size, filter_size)\
                      * 2*weight_limit - weight_limit
        #self.bias = np.zeros(out_chan)
        self.bias = np.random.rand(out_chan) * 2*weight_limit - weight_limit

        self._reset_gradients()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):   # x SHAPE : (B, C_in, H_in, W_in)
        self._calculate_out_shape(x)

		# apply padding to input x
        if self.padding != 0:
            pad_h = x.shape[2] + 2*self.padding
            pad_w = x.shape[3] + 2*self.padding
            x_pad = np.zeros((x.shape[0], x.shape[1], pad_h, pad_w))
            x_pad[:, :,
                  self.padding:-self.padding,
                  self.padding:-self.padding] = x
            x = x_pad

        # flat_x SHAPE : (H_out*W_out*B, C_in*F*F)
        self.flat_x = self._flatten_conv(x, self.stride)
        # weight_flat SHAPE : (C_out, C_in*F*F)
        weight_flat = self.weight.reshape(self.weight.shape[0], -1)

        # conv_flat SHAPE : (H_out*W_out*B, C_out)
        conv_flat = np.dot(self.flat_x, weight_flat.T) + self.bias
        # conv SHAPE: (C_out, H_out, W_out, B)
        conv = conv_flat.T.reshape(*self.shape_out[1:], self.shape_out[0])
        # conv SHAPE: (B, C_out, H_out, W_out)
        conv = conv.transpose(3,0,1,2)

        return conv

    def gradient(self, prev_grad):
        # if gradient comes from linear layer SHAPE : (B, C_out*H_out*W_out)
        if prev_grad.shape != self.shape_out:
             # prev_grad SHAPE : (B, C_out, H_out, W_out)
            prev_grad = prev_grad.reshape(self.shape_out)

        # prev_grad_flat SHAPE : (C_out, H_out*W_out*B)
        prev_grad_flat = prev_grad.transpose(1,2,3,0).reshape(
                                                        self.shape_out[1], -1)

        # calculate gradients for weights and biases
        self.w_grad += np.dot(prev_grad_flat, self.flat_x).reshape(
                                                          self.weight.shape)
        self.b_grad += np.sum(prev_grad, axis=(0,2,3))

        # calculate gradients of the input by transposed convolution
        next_grad = self._transposed_conv(prev_grad)

        return next_grad # return SHAPE : (B, C_in, H_in, W_in)

    def _reset_gradients(self):
        self.w_grad = np.zeros(self.weight.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def _calculate_out_shape(self, x):
        h_out = int((x.shape[2] - self.filter_size + 2*self.padding)
                     //self.stride+1)
        w_out = int((x.shape[3] - self.filter_size + 2*self.padding)
                     //self.stride+1)

        self.shape_out = (x.shape[0], self.out_chan, h_out, w_out)
        self.shape_in = x.shape

    def _flatten_conv(self, x, stride,
                      h_out=None, w_out=None):  # x SHAPE : (B, C, H, W)
		# flattens the convolution to a 2D-Array
		# of the shape (H_out*W_out*B, C_in*F*F)

        if h_out == None and w_out == None:
            h_out, w_out = self.shape_out[2], self.shape_out[3]

        # flatten_x SHAPE : (H_out*W_out*B, C_in*F*F)
        flatten_x = np.zeros((h_out*w_out*x.shape[0],
                              x.shape[1]*self.filter_size*self.filter_size))

        for h in range(0, h_out):
            flatten_h = h*w_out*x.shape[0]
            for w in range(0, w_out):
                flatten_w = w*x.shape[0]
                step_size = flatten_h+flatten_w
                flatten_x[step_size:step_size+x.shape[0],:] =\
                        x[:, :,
                          h*stride:h*stride+self.filter_size,
                          w*stride:w*stride+self.filter_size].reshape(
                                                                x.shape[0],-1)
        return flatten_x

    def _transposed_conv(self, x):  # x SHAPE : (B, C_out, H_out, W_out)
        # more infos about transposed conv: https://arxiv.org/pdf/1603.07285.pdf
        # and animations: https://github.com/vdumoulin/conv_arithmetic

		# apply stride to the input x
        if self.stride != 1:
            stride_h = self.stride*(x.shape[2])
            stride_w  = self.stride*(x.shape[3])
            x_stride = np.zeros((*x.shape[:2], stride_h, stride_w))
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x_stride[:, :, h*self.stride, w*self.stride] = x[:, :, h, w]
            x = x_stride

		# apply padding to the input x
        padding = self.filter_size-1 - self.padding
        if padding == 0:
            pass
        elif padding < 0:
            padding = abs(padding)
            x = x[:, :, padding:-padding, padding:-padding]
        else:
            x_pad = np.zeros((x.shape[0], x.shape[1],
                              x.shape[2]+2*padding, x.shape[3]+2*padding))
            x_pad[:, :, padding:-padding, padding:-padding] = x
            x = x_pad

        # x_flat SHAPE : (H_in*W_in*B, C_out*F*F)
        x_flat = self._flatten_conv(x, 1, *self.shape_in[2:])
        # rotate weights 180 degrees
		# animation about why the weights need to be rotated
		# https://miro.medium.com/max/986/1*HnxnJDq-IgsSS0q3Lut4xA.gif
        weight_rot = np.flip(self.weight, (2,3))
        # weights_rot_flat SHAPE : (C_in, C_out*F*F)
        weight_rot_flat = weight_rot.transpose(1,0,2,3).reshape(
                                                        self.weight.shape[1],-1)

        # trans_conv_flat SHAPE : (H_in*W_in*B, C_in)
        trans_conv_flat = np.dot(x_flat, weight_rot_flat.T)
        # trans_conv SHAPE: (C_in, H_in, W_in, B)
        trans_conv = trans_conv_flat.T.reshape(*self.shape_in[1:],
                                                self.shape_in[0])
        # conv SHAPE: (B, C_in, H_in, W_in)
        trans_conv = trans_conv.transpose(3,0,1,2)

        return trans_conv
