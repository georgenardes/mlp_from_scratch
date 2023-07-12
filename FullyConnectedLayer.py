import numpy as np
from quantizer import quantize, stochastic_rounding
import cupy as cp

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
        # escala da escala de gradiente
        self.grad_output_scale = 1
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

    
    def backward_with_scale(self, grad_output, grad_scale, learning_rate):
        """ grad_output é o erro que chega para esta camada """

        # scaling gradients        
        grad_output = (grad_output / self.output_scale) * (self.weights_scale * self.input_scale) * (grad_scale)
        
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)

        # para simular a operação em hardware, será necessário salvar o grad_weights escalado e quantizado 
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases

        # scale de grad_output or grad_of_input
        self.grad_output_scale = np.max(np.abs(grad_output))
        grad_input = grad_input / self.grad_output_scale

        return grad_input





class QFullyConnectedLayerWithScale:
    """ This is a FC layer with scale and SX4 Quantization"""
    

    def __init__(self, input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.qw = None # quantized weight use for backprop
        self.biases = np.zeros((1, output_size))

        #################################################
        # escala inicial dos pesos
        self.weights_scale = np.max(np.abs(self.weights))
        self.input_scale = None
        self.output_scale = 1
        # escala de gradiente
        self.grad_output_scale = 1

        # escala de gradiente dos pesos
        self.grad_weights_scale = 1

        #################################################


    def forward(self, inputs):
        """ Default forward """

        self.inputs = inputs        
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output
    
    
    def foward_with_scale(self, inputs, x_scale):
        # salva entrada para backprop, (entrada já vem quantizada)
        self.inputs = inputs

        # salva escala de entrada
        self.input_scale = x_scale
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = np.max(np.abs(self.weights))
        # escala os pesos 
        w = self.weights / self.weights_scale
        # escala os biases com base na escala dos pesos e escala das ativações
        b = self.biases / (self.weights_scale * self.input_scale)

        # quantiza os pesos
        self.qw = quantize(w, True)
        # quantiza os biases
        self.qb = quantize(b, True)   
                     
        #################################################

        # faz matmul e desescala pesos e biases                
        self.output = (np.matmul(inputs, self.qw) + self.qb) * (self.weights_scale * self.input_scale)
        
        # descobre escala da saída com base em uma média
        self.output_scale = 0.9 * self.output_scale + 0.1 * np.max(np.abs(self.output))      

        # escala saída
        self.output = self.output / self.output_scale

        # quantiza saída
        self.output = quantize(self.output, True)
        #################################################

        return self.output


    def backward(self, grad_output, learning_rate):
        
        # scaling gradients        
        grad_output = (grad_output ) * (self.weights_scale * self.input_scale / self.output_scale)
        
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases
        return grad_input

    
    def backward_with_scale(self, grad_output, grad_scale, learning_rate):
        """ grad_output é o erro que chega para esta camada (grad output já vem quantizado)"""        
        
        # scaling gradients        
        grad_output = (grad_output) * (self.weights_scale * self.input_scale * grad_scale  / self.output_scale)         

        # gradient calculation                 
        grad_input = np.matmul(grad_output, self.qw.T)

        # para simular a operação em hardware, será necessário salvar o grad_weights escalado e quantizado 
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # get the grad w scale
        grad_weights_scale = np.max(np.abs(grad_weights))
        grad_bias_scale = np.max(np.abs(grad_biases))
        # scale the grad
        grad_weights /= grad_weights_scale
        grad_biases /= grad_bias_scale
        # quantize the grad
        qgw = quantize(grad_weights, True)
        qgb = quantize(grad_biases, True)
        # scale back the grad
        grad_weights = qgw * grad_weights_scale
        grad_biases = qgb * grad_bias_scale

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases

        # scale de grad_output or grad_of_input
        self.grad_output_scale =  0.9 * self.grad_output_scale + 0.1 * np.max(np.abs(grad_input))
        grad_input = grad_input / self.grad_output_scale

        # quantiza o gradiente
        grad_input = quantize(grad_input, True)

        return grad_input
    
    def backward_with_scale_and_quantization(self, grad_output, grad_scale, learning_rate):
        """ grad_output é o erro que chega para esta camada """        

        # scaling gradients        
        grad_output = (grad_output) * (self.weights_scale * self.input_scale * grad_scale  / self.output_scale)         

        # gradient calculation 
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)

        # para simular a operação em hardware, será necessário salvar o grad_weights escalado e quantizado 
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases

        # scale de grad_output or grad_of_input
        self.grad_output_scale =  0.9 * self.grad_output_scale + 0.1 * np.max(np.abs(grad_input))
        grad_input = grad_input / self.grad_output_scale

        # quantiza o gradiente
        grad_input = quantize(grad_input, True)

        return grad_input