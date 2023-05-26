from keras.datasets import mnist
import numpy as np
from NN import NeuralNetwork
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 784) / 255.
X_test = X_test.reshape(-1, 784) / 255.

# Convert labels to one-hot vectors
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Create an instance of the NeuralNetwork class
model = NeuralNetwork(input_size=784, output_size=10, hidden_layers=[120, 120], activation='relu')

# Train the model
model.train(X_train, y_train, epochs=70, batch_size=128, lr=0.003, optimizer='adam')

# Test the model
predictions = model.predict(X_test)
accuracy = model.accuracy(y_test, predictions)

print("Accuracy on test set: {:.2%}".format(accuracy))
