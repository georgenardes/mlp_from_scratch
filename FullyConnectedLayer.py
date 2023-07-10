import numpy as np


class FullyConnectedLayer:
    """ this is a vanilla FC layer """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate):
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T)
        grad_weights = np.matmul(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input
    


class FullyConnectedLayerWithScale:
    """ This is a FC layer with scale """
    

    def __init__(self, input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.biases = np.zeros((1, output_size))

        #################################################
        # escala inicial dos pesos
        self.weights_scale = np.max(np.abs(self.weights))
        self.input_scale = None
        self.output_scale = 1

        #################################################


    def forward(self, inputs):
        """ Default forward """

        self.inputs = inputs        
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output
    
    
    def foward_with_scale(self, inputs, x_scale):

        # salva escala de entrada
        self.input_scale = x_scale
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = np.max(np.abs(self.weights))
        # escala os pesos 
        w = self.weights / self.weights_scale
        # escala os biases com base na escala dos pesos e escala das ativações
        b = self.biases / (self.weights_scale * self.input_scale)
        #################################################

        # faz matmul e desescala pesos e biases        
        self.inputs = inputs
        self.output = (np.matmul(inputs, w) + b) * (self.weights_scale * self.input_scale)
        
        # descobre escala da saída com base em uma média
        self.output_scale = 0.9 * self.output_scale + 0.1 * np.max(np.abs(self.output))      

        # escala saída
        self.output = self.output / self.output_scale
        #################################################

        return self.output


    def backward(self, grad_output, learning_rate):
        
        # scaling gradients        
        grad_output = (grad_output / self.output_scale) * (self.weights_scale * self.input_scale)
        
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases
        return grad_input




class QFullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs

        #
        # APPLY QUANTIZATION ON THE WEIGHTS AND BIASES HERE
        #

        qweights = self.weight_quantization(self.weights)
        self.output = np.matmul(inputs, qweights) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_input = np.matmul(grad_output, self.weights.T)
        grad_weights = np.matmul(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
    

    def weight_quantization(self, weights):
        qweights = weights
        return qweights 
    
