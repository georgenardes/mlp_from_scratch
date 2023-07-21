import numpy as np
import cupy as cp
import tensorflow as tf
from quantizer import quantize

class ActivationLayer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError


class ReLU(ActivationLayer):
    def __init__(self):
        self.relu = tf.keras.layers.ReLU()

    def forward(self, inputs):
        self.inputs = inputs        
        return self.relu(inputs)

    def backward(self, dz, learning_rate, **kwargs):
        dx = dz * tf.where(self.inputs > 0., 1., 0.)        
        return dx
    

class QReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        out = tf.maximum(0, inputs)
        out = quantize(out, stochastic_round=True, stochastic_zero=True)
        return out

    def backward(self, grad_output, learning_rate):
        return grad_output * tf.where(self.inputs > 0, 1., 0.)


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
        self.outputs = tf.exp(inputs - tf.reduce_max(inputs, axis=-1, keepdims=True))
        self.outputs /= tf.reduce_sum(self.outputs, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, dz, learning_rate):
        return dz