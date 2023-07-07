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

        # escala os pesos com base na escala
        self.weights = self.weights / self.weights_scale
        self.biases = self.biases / self.weights_scale # desnecessário pq o bias é zero
        #################################################


    def forward(self, inputs):
        """ Default forward """

        self.inputs = inputs        
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output
    
    
    def foward_with_scale(self, inputs, x_scale):

        # salva escala de entrada
        self.input_scale = x_scale
        # descobre a escala dos pesos com base na média
        self.weights_scale = np.max(np.abs(self.weights))
        # escala os pesos 
        self.weights = self.weights / self.weights_scale
        # escala os biases com base na escala dos pesos e escala das ativações
        self.biases = self.biases / (self.weights_scale * self.input_scale)
        #################################################

        # faz matmul
        self.output = self.forward(inputs) 

        # desescala a saída 
        self.output = self.output * (self.weights_scale * self.input_scale)

        # descobre escala da saída
        self.output_scale = 0.9 * self.output_scale + 0.1 * np.max(np.abs(self.output))      

        # escala saída
        self.output = self.output / self.output_scale
        #################################################

        return self.output


    def backward(self, grad_output, learning_rate):
        

        ## FOI ALCANÇADO 91% DE ACURÁCIA ## SEM O ESCALA DO OUTPUT
        ## FOI ALCANÇADO 94% COM A ESCALA --->> grad_output = (grad_output / self.output_scale) *  (self.weights_scale * self.input_scale)
        grad_output = (grad_output / self.output_scale) *  (self.weights_scale * self.input_scale)

        # desescala pesos # desescala bias
        self.weights = self.weights * self.weights_scale
        self.biases = self.biases * self.weights_scale * self.input_scale

        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T)
        grad_weights = np.matmul(self.inputs.T * self.input_scale, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

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
    
