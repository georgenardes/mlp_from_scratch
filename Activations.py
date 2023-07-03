import numpy as np


class ActivationLayer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError

class ReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output, learning_rate):
        return grad_output * np.where(self.inputs > 0, 1, 0)
    
class QReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output, learning_rate):
        return grad_output * np.where(self.inputs > 0, 1, 0)


class Sigmoid(ActivationLayer):
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output * self.outputs * (1 - self.outputs)

class Tanh(ActivationLayer):
    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output * (1 - self.outputs**2)
    
    
class Softmax(ActivationLayer):
    def forward(self, inputs):
        self.outputs = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.outputs /= np.sum(self.outputs, axis=-1, keepdims=True)
        return self.outputs

    def backward(self, grad_output, learning_rate):
        return grad_output