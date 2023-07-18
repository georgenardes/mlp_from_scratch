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

        w = cp.random.randn(input_size, output_size) * cp.sqrt(2/input_size) 
        self.weights_scale = cp.max(cp.abs(w))
        self.qw = quantize(w/self.weights_scale, True, True)

        b = cp.zeros((1, output_size))
        self.bias_scale = 1
        self.qb = quantize(b, True, False) # quantized bias

        #################################################
        # escala inicial dos pesos
        self.ws_hist = []
        self.bs_hist = []
        self.input_scale = None
        self.output_scale = 1
        self.os_hist = []
        
        # escala de gradiente
        self.grad_output_scale = 1
        self.gos_hist = []
        self.grad_output_hist = []

        # escala de gradiente dos pesos
        self.grad_weights_scale = 1
        self.gws_hist = []
        self.grad_bias_scale = 1 
        self.gbs_hist = []
        #################################################
        
        self.update_scale = True

    
    
    def foward_with_scale(self, inputs, x_scale):
        # salva entrada para backprop, (entrada já vem quantizada)
        self.inputs = inputs

        # salva escala de entrada
        self.input_scale = x_scale                            

        # faz matmul e desescala pesos e biases                
        self.output = (cp.matmul(inputs, self.qw) + self.qb) * (self.weights_scale * self.input_scale)
                
        # descobre escala da saída com base em uma média
        self.output_scale = 0.99 * self.output_scale + 0.01 * cp.max(cp.abs(self.output))      
        self.os_hist.append(self.output_scale)

        # escala saída
        self.output = self.output / self.output_scale

        # quantiza saída
        self.output = quantize(self.output, stochastic_round=True, stochastic_zero=True)
        #################################################

        return self.output

    
    def backward_with_scale(self, grad_output, grad_scale, learning_rate):
        """ 
        Esta função faz propagação do erro para os pesos e para a entrada.

        grad_output: esse parâmetro é o Erro que vem da camada l+1. Ele vem quantizado para Deep Nibble.
        grad_scale: esse é a Escala usada para normalizar o Erro antes de quantiza-lo para Deep Nibble
        learning_rate: taxa de aprendizado
        """                

        self.gos_hist.append(grad_scale)
        
        # if 1000 < self.iteration < 11000:
        #   self.grad_output_hist.append(cp.asnumpy(grad_output))

        # scaling gradients                
        grad_output = grad_output * (self.weights_scale * self.input_scale * grad_scale  / self.output_scale)    
        
        # gradient calculation. self.qw é a matriz de pesos quantizados utilizados na forward prop
        grad_input = cp.matmul(grad_output, self.qw.T) 

        # scale de grad_output or grad_of_input
        self.grad_output_scale =  0.9 * self.grad_output_scale + 0.1 * cp.max(cp.abs(grad_input))
        grad_input = grad_input / self.grad_output_scale

        # quantiza o gradiente
        grad_input = quantize(grad_input, stochastic_round=True, stochastic_zero=True)
                
        # calcula o gradiente dos pesos. self.inputs é a entrada dessa camada na etapa de forward prop. Ela é quantizada. 
        # self.weights_scale é a escala utilizada para os pesos na etapa forward prop e é necessário aqui por conta da derivada
        grad_weights = cp.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = cp.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # get the grad w scale
        self.grad_weights_scale = 0.9 * self.grad_weights_scale + 0.1 * cp.max(cp.abs(grad_weights))        
        self.gws_hist.append(self.grad_weights_scale)

        # get the grad b scale
        self.grad_bias_scale = 0.9 * self.grad_bias_scale + 0.1 * cp.max(cp.abs(grad_biases))        
        self.gbs_hist.append(self.grad_bias_scale)

        # scale the grad
        grad_weights /= self.grad_weights_scale
        grad_biases /= self.grad_bias_scale
        
        # quantize the grad
        qgw = quantize(grad_weights, True)
        qgb = quantize(grad_biases, True)        


        #################### ETAPA DE ATUALIZAÇÃO DOS PESOS #######################
        # weight scaling
        self.qw = self.qw * self.weights_scale
        # gradient scaling
        qgw = qgw * self.grad_weights_scale
        # weight updating
        self.qw = self.qw - learning_rate * qgw
        
        # bias scaling
        self.qb = self.qb * (self.weights_scale * self.input_scale) 
        # bias gradient scaling
        qgb = qgb * self.grad_bias_scale
        # bias updating
        self.qb = self.qb - learning_rate * qgb
        
        ############################################################################
        # ############ ETAPA DE CLIP, ESCALA E QUANTIZAÇÃO ################

        # atribui a weights. Weights será escalado e quantizado durante inferência
        w = self.qw 
        w = cp.clip(w, -7, 7)         
        b = self.qb
        b = cp.clip(b, -127., 127.) 
            
        # colocar quantização aqui e remover do forward
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = 0.99*self.weights_scale + 0.01*cp.max(cp.abs(w))
        self.ws_hist.append(self.weights_scale)

        self.bias_scale = self.weights_scale * self.input_scale
        self.bs_hist.append(self.bias_scale)
        
        # escala os pesos 
        w = w / self.weights_scale
        
        # escala os biases 
        b = b / self.bias_scale
        
        
        # quantiza peesos e bias
        self.qw = quantize(w, True, True)
        self.qb = quantize(b, True, False)
               
        
        return grad_input
    
    