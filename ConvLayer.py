import tensorflow as tf
import numpy as np
from quantizer import quantize, quantize_po2

class ConvLayer():
    def __init__(self, nfilters, kernel_size, input_channels, strides=[1,1,1,1], padding='SAME'):        
        # glorot_initializer = tf.keras.initializers.GlorotNormal()
        # self.filters = glorot_initializer((kernel_size, kernel_size, input_channels, nfilters), tf.float32)
        
        # he_initializer = tf.keras.initializers.HeUniform()
        # self.filters = he_initializer((kernel_size, kernel_size, input_channels, nfilters), tf.float32)

        # esse inicializador funcionou melhor na rede 32 bits
        self.filters = tf.constant(np.random.randn(kernel_size, kernel_size, input_channels, nfilters) * np.sqrt(2/input_channels), dtype=tf.float32)
        self.bias = tf.constant(np.zeros((1,1,1,nfilters)), dtype=tf.float32)
        
        # other atributes        
        self.strides = strides
        self.padding = padding
                
        
    def forward(self, inputs):
        self.inputs = inputs
        z =  tf.add(tf.nn.conv2d(input=inputs, filters=self.filters, strides=self.strides, padding=self.padding), self.bias)
        return z


    def backward(self, dz, learning_rate):
        dx = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(self.inputs), self.filters, dz, strides=self.strides, padding=self.padding)
        dw = tf.compat.v1.nn.conv2d_backprop_filter(self.inputs, tf.shape(self.filters), dz, strides=self.strides, padding=self.padding)
        db = tf.reduce_sum(dz, (0,1,2), True)

        # update weights
        self.filters = self.filters - learning_rate * dw
        self.bias = self.bias - learning_rate * db

        return dx



class QConvLayer():
    def __init__(self, nfilters, kernel_size, input_channels, strides=[1,1,1,1], padding='SAME'):        
        # glorot_initializer = tf.keras.initializers.GlorotNormal()
        # self.filters = glorot_initializer((kernel_size, kernel_size, input_channels, nfilters), tf.float32)
        
        # he_initializer = tf.keras.initializers.HeUniform()
        # self.filters = he_initializer((kernel_size, kernel_size, input_channels, nfilters), tf.float32)

        # esse inicializador funcionou melhor na rede 32 bits
        w = tf.constant(np.random.randn(kernel_size, kernel_size, input_channels, nfilters) * np.sqrt(2/input_channels), dtype=tf.float32)        
        self.weights_scale = tf.reduce_max(tf.abs(w))
        self.qw = quantize(w/self.weights_scale, True, True)
        
        b = tf.constant(np.zeros((1,1,1,nfilters)), dtype=tf.float32)
        self.qb = quantize(b, True, False) # quantized bias
    

        #################################################
        # escala inicial dos pesos
        self.ws_hist = []
        self.bs_hist = []
        self.input_scale = None
        self.output_scale = tf.constant(1, tf.float32)
        self.os_hist = []
        
        # escala de gradiente
        self.grad_output_scale = tf.constant(1, tf.float32)
        self.gos_hist = []
        self.grad_output_hist = []

        # escala de gradiente dos pesos
        self.grad_weights_scale = tf.constant(1, tf.float32)
        self.gws_hist = []
        self.grad_bias_scale = tf.constant(1, tf.float32)
        self.gbs_hist = []
        #################################################        

        # other atributes        
        self.strides = strides
        self.padding = padding


        
    def qforward(self, inputs, xs):
        # salva entrada para backprop, (entrada já vem quantizada)
        self.inputs = inputs

        # salva escala de entrada
        self.input_scale = xs 
        qxs = quantize_po2(self.input_scale)
        qws = quantize_po2(self.weights_scale)

        # faz matmul e desescala pesos e biases
        self.output = (tf.nn.conv2d(input=inputs, filters=self.qw, strides=self.strides, padding=self.padding) + self.qb) * (qws * qxs)

        # descobre escala da saída com base em uma média
        self.output_scale = 0.99 * self.output_scale + 0.01 * tf.reduce_max(tf.abs(self.output))
        qos = quantize_po2(self.output_scale)
        
        self.os_hist.append(qos)

        # escala saída
        self.output = self.output / qos

        # quantiza saída
        self.output = quantize(self.output, stochastic_round=True, stochastic_zero=True)
        #################################################

        return self.output


    def qbackward(self, dz, grad_scale, learning_rate):
        """ 
        Esta função faz propagação do erro para os pesos e para a entrada.

        dz: esse parâmetro é o Erro que vem da camada l+1. Ele vem quantizado para Deep Nibble.
        grad_scale: esse é a Escala usada para normalizar o Erro antes de quantiza-lo para Deep Nibble
        learning_rate: taxa de aprendizado
        """         
        
        qws = quantize_po2(self.weights_scale)
        qxs = quantize_po2(self.input_scale)
        qgos = quantize_po2(grad_scale)        
        qos = quantize_po2(self.output_scale)

        self.gos_hist.append(qgos) 

        # gradient calculation. self.qw é a matriz de pesos quantizados utilizados na forward prop
        dx = tf.compat.v1.nn.conv2d_backprop_input(tf.shape(self.inputs), self.qw, dz, strides=self.strides, padding=self.padding) * (qws * qxs * qgos  / qos)
        # scale de grad_output or grad_of_input
        self.grad_output_scale =  0.9 * self.grad_output_scale + 0.1 * tf.reduce_max(tf.abs(dx))
        qgis = quantize_po2(self.grad_output_scale)
        dx = dx / qgis
        # quantiza o gradiente
        qdx = quantize(dx, stochastic_round=True, stochastic_zero=True)
        
        # calcula o gradiente dos pesos. self.inputs é a entrada dessa camada na etapa de forward prop. Ela é quantizada.             
        dw = tf.compat.v1.nn.conv2d_backprop_filter(self.inputs, tf.shape(self.qw), dz, strides=self.strides, padding=self.padding) * (qxs * qgos / qos) 
        db = tf.reduce_sum(dz, (0,1,2), True) * (qgos / qos) 

        # get the grad w scale
        self.grad_weights_scale = 0.9 * self.grad_weights_scale + 0.1 * tf.reduce_max(tf.abs(dw))       
        qgws = quantize_po2(self.grad_weights_scale)
        self.gws_hist.append(qgws)        

        # get the grad b scale
        self.grad_bias_scale = 0.9 * self.grad_bias_scale + 0.1 * tf.reduce_max(tf.abs(db))        
        qgbs = quantize_po2(self.grad_bias_scale)
        self.gbs_hist.append(qgbs)

        # scale the grad
        dw /= qgws
        db /= qgbs
        
        # quantize the grad
        qgw = quantize(dw, True)
        qgb = quantize(db, True)                

        #################### ETAPA DE ATUALIZAÇÃO DOS PESOS #######################
        # weight scaling
        self.qw = self.qw * qws
        # gradient scaling
        qgw = qgw * qgws
        # weight updating
        self.qw = self.qw - learning_rate * qgw
        
        # bias scaling
        self.qb = self.qb * (qws * qxs) 
        # bias gradient scaling
        qgb = qgb * qgbs
        # bias updating
        self.qb = self.qb - learning_rate * qgb

        ############################################################################
        # ############ ETAPA DE CLIP, ESCALA E QUANTIZAÇÃO ################

        # atribui a weights. Weights será escalado e quantizado durante inferência
        w = self.qw 
        w = tf.clip_by_value(w, -7, 7)         
        b = self.qb
        b = tf.clip_by_value(b, -127., 127.) 
            
        # colocar quantização aqui e remover do forward
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = 0.99*self.weights_scale + 0.01*tf.reduce_max(tf.abs(w))
        qws = quantize_po2(self.weights_scale)
        self.ws_hist.append(qws)
                
        # escala os pesos 
        w = w / qws
        
        # escala os biases 
        self.bs_hist.append(qws * qxs)
        b = b / (qws * qxs)
        
        
        # quantiza peesos e bias
        self.qw = quantize(w, True, True)
        self.qb = quantize(b, True, False)        

        return qdx




class CustomMaxPool():
    def __init__(self, ksize=2, stride=(2,2)):
        self.ksize = ksize
        self.stride = stride
        self.upsampler = tf.keras.layers.UpSampling2D(size=ksize)
        self.maxpooler = tf.keras.layers.MaxPool2D(pool_size=ksize, strides=stride, padding='VALID')


    def forward(self, inputs):
        self.inputs = inputs
        z = self.maxpooler(inputs)
        return z

    def backward(self, dz, *kwargs):
        dx = self.upsampler(dz)
        return dx
    

class CustomFlatten():
    def __init__(self, input_shape):
        self.input_shape = input_shape        
        self.flattener = tf.keras.layers.Flatten()
        self.reshapener = tf.keras.layers.Reshape(input_shape)

    def forward(self, inputs):
        return self.flattener(inputs)
    
    def backward(self, dz, *wargs):
        return self.reshapener(dz)