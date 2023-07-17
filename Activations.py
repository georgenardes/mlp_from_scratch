import numpy as np
import cupy as cp
from quantizer import quantize

class ActivationLayer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError

class ReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return cp.maximum(0, inputs)

    def backward(self, grad_output, learning_rate, **kwargs):
        return grad_output * cp.where(self.inputs > 0, 1, 0)
    
class QReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        out = cp.maximum(0, inputs)
        out = quantize(out, stochastic_round=True, stochastic_zero=True)
        return out

    def backward(self, grad_output, learning_rate):
        return grad_output * cp.where(self.inputs > 0, 1, 0)


class Sigmoid(ActivationLayer):
    def forward(self, inputs):
        self.outputs = 1 / (1 + cp.exp(-inputs))
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output * self.outputs * (1 - self.outputs)

class Tanh(ActivationLayer):
    def forward(self, inputs):
        self.outputs = cp.tanh(inputs)
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output * (1 - self.outputs**2)
    
    
class Softmax(ActivationLayer):
    def forward(self, inputs):
        self.outputs = cp.exp(inputs - cp.max(inputs, axis=-1, keepdims=True))
        self.outputs /= cp.sum(self.outputs, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output