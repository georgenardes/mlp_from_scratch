import tensorflow as tf
import numpy as np


class CustomConvLayer():
    def __init__(self, nfilters, kernel_size, input_channels, strides=[1,1,1,1], padding='SAME'):        
        
        self.filters = tf.constant(np.random.randn(kernel_size, kernel_size, input_channels, nfilters), dtype=tf.float32)
        self.bias = tf.constant(np.random.randn(1,1,1,nfilters), dtype=tf.float32)
        
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