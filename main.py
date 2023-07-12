import numpy as np
import tensorflow as tf
from dataset import load_mnist
from NeuralNetwork import QNeuralNetworkWithScale
import Activations
import FullyConnectedLayer


# load dataset
x_train, x_test, y_train, y_test = load_mnist()

# Define neural network inputnoutput
input_size = x_train.shape[1]
output_size = y_train.shape[1]

print(x_train.shape,
x_test.shape,
y_train.shape,
y_test.shape)

# Create and train the neural network
neural_network_with_scale = QNeuralNetworkWithScale(input_size, output_size)

# train the nn
neural_network_with_scale.train(x_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=64)

