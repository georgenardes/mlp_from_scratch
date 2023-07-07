import numpy as np
from FullyConnectedLayer import FullyConnectedLayer, FullyConnectedLayerWithScale
from Activations import *


class NeuralNetwork:
    """ vanilla NN """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        self.layers.append(FullyConnectedLayer(input_size, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, output_size))

        self.softmax = Softmax()
    

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output


    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


    def train(self, inputs, targets, learning_rate, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)

                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")


    def predict(self, inputs):
        outputs = []
        for input in inputs:
            output = self.forward(input)
            predicted_class = np.argmax(output)
            outputs.append(predicted_class)
        return np.array(outputs)


    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]
        loss = np.sum(-targets * np.log(output + 1e-8)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.
        grad_output = output - targets
        return grad_output


    def get_batches(self, inputs, targets, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            num_batches = len(inputs) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                yield inputs[start:end], targets[start:end]
            if len(inputs) % batch_size != 0:
                yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]



class NeuralNetworkWithScale:
    """ rede neural com tratamento de escala de pesos e ativações """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        self.layers.append(FullyConnectedLayerWithScale(input_size, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayerWithScale(256, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayerWithScale(256, output_size))

        self.softmax = Softmax()
    

    def forward(self, inputs):
        # descobre a escala do dado de entrada
        x_scale = np.max(np.abs(inputs))    
        
        # escala entrada e atribui a variavel output que entrará no laço
        output = inputs / x_scale          

        for layer in self.layers:

            if isinstance(layer, FullyConnectedLayerWithScale):
                output = layer.foward_with_scale(output, x_scale=x_scale)
                x_scale = layer.output_scale

            else:
                output = layer.forward(output)

        # desnormaliza saída
        output = output * x_scale

        return output


    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


    def train(self, inputs, targets, learning_rate, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)                
                
                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")


    def predict(self, inputs):
        outputs = []
        for input in inputs:
            output = self.forward(input)
            predicted_class = np.argmax(output)
            outputs.append(predicted_class)
        return np.array(outputs)


    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]
        loss = np.sum(-targets * np.log(output + 1e-8)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.
        grad_output = output - targets
        return grad_output


    def get_batches(self, inputs, targets, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            num_batches = len(inputs) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                yield inputs[start:end], targets[start:end]
            if len(inputs) % batch_size != 0:
                yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]                

                