import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork



# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0
y = mnist.target

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define neural network architecture
input_size = X_train.shape[1]
output_size = y_train.shape[1]

# Create and train the neural network
neural_network = NeuralNetwork(input_size, output_size)
neural_network.train(X_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=64)

# Make predictions on the test set
y_pred = neural_network.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"Accuracy: {accuracy * 100}%")
