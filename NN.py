import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers=[], hidden_size=4, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)


        

    def adam_update(self, gradients, parameters, ms, vs, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m, v, param = ms, vs, parameters
        dw, db = gradients

        # Update biased first moment estimate
        m[0] = beta1 * m[0] + (1 - beta1) * dw
        m[1] = beta1 * m[1] + (1 - beta1) * db

        # Update biased second raw moment estimate
        v[0] = beta2 * v[0] + (1 - beta2) * dw**2
        v[1] = beta2 * v[1] + (1 - beta2) * db**2

        # Compute bias-corrected first moment estimate
        m_hat = [mi / (1 - beta1**(self.t+1)) for mi in m]

        # Compute bias-corrected second raw moment estimate
        v_hat = [vi / (1 - beta2**(self.t+1)) for vi in v]

        # Update parameters
        param[0] -= lr * m_hat[0] / (np.sqrt(v_hat[0]) + eps)
        param[1] -= lr * m_hat[1] / (np.sqrt(v_hat[1]) + eps)

        return param, m, v

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, y, y_hat):
        n_samples = y.shape[0]
        logp = - np.log(y_hat[np.arange(n_samples), y.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def forward(self, X):
        activations = []
        for i in range(len(self.weights)):
            if i == 0:
                Z = np.dot(X, self.weights[i]) + self.biases[i]
            else:
                Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                activation = self.softmax(Z)
            elif self.activation == 'relu':
                activation = self.relu(Z)
            else:
                activation = self.sigmoid(Z)
            activations.append(activation)
        return activations
    
    def backprop(self, X, y, activations):
      dweights = []
      dbiases = []
      dZ = activations[-1] - y  # error from the output layer
      for i in range(len(self.weights) - 1, -1, -1):
          dW = np.dot(activations[i-1].T, dZ) if i > 0 else np.dot(X.T, dZ)
          db = np.sum(dZ, axis=0, keepdims=True)
          dweights.insert(0, dW)
          dbiases.insert(0, db)
          if i > 0:  # Compute dZ for the next iteration
              dZ = np.dot(dZ, self.weights[i].T) * (activations[i-1] > 0).astype(int)
      return dweights, dbiases



    
    def train(self, X, y, epochs=100, lr=0.1, batch_size=32, optimizer='adam'):
      losses = []

      # Initialize parameters for Adam
      m_weights = [np.zeros_like(w) for w in self.weights]
      v_weights = [np.zeros_like(w) for w in self.weights]

      m_biases = [np.zeros_like(b) for b in self.biases]
      v_biases = [np.zeros_like(b) for b in self.biases]

      beta1 = 0.9
      beta2 = 0.999
      eps = 1e-8

      for epoch in range(epochs):
          # Mini-Batch Gradient Descent
          indices = np.arange(X.shape[0])
          np.random.shuffle(indices)
          X = X[indices]
          y = y[indices]

          for start in range(0, X.shape[0], batch_size):
              end = min(start + batch_size, X.shape[0])
              X_batch, y_batch = X[start:end], y[start:end]
              
              activations = self.forward(X_batch)
              dweights, dbiases = self.backprop(X_batch, y_batch, activations)

              for i in range(len(self.weights)):
                  dw = dweights[i]
                  db = dbiases[i]

                  if optimizer == 'adam':
                      # Compute Adam updates for weights
                      m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * dw
                      v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * np.square(dw)
                      m_hat = m_weights[i] / (1 - np.power(beta1, epoch+1))  # bias correction
                      v_hat = v_weights[i] / (1 - np.power(beta2, epoch+1))
                      self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

                      # Compute Adam updates for biases
                      m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * db
                      v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * np.square(db)
                      m_hat = m_biases[i] / (1 - np.power(beta1, epoch+1))  # bias correction
                      v_hat = v_biases[i] / (1 - np.power(beta2, epoch+1))
                      self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                  
                  else:
                      # Standard Gradient Descent
                      self.weights[i] -= lr * dw
                      self.biases[i] -= lr * db

          # Compute loss and add it to the list of losses
          activations = self.forward(X)
          loss = self.compute_loss(y, activations[-1])
          losses.append(loss)

          # Print loss every 10 epochs
          if epoch % 10 == 0:
              print(f'Epoch: {epoch}, Loss: {loss}')
      
      return losses



    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, y_true, y_pred):
        return accuracy_score(np.argmax(y_true, axis=1), y_pred)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)
    
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.weights, self.biases = pickle.load(file)

    def tuner(self, X, y, hidden_layers=[], hidden_size_range=(2, 10), epochs_range=(10, 100), lr_range=(0.001, 1)):
        best_acc = 0
        best_model = None
        
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for h in hidden_layers:
            for hs in range(hidden_size_range[0], hidden_size_range[1]+1):
                for lr in np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num=5):
                    for epochs in range(epochs_range[0], epochs_range[1]+1):
                        # Train model with current hyperparameters
                        model = NeuralNetwork(X.shape[1], y.shape[1], hidden_layers=[hs]*h)
                        losses = model.train(X_train, y_train, epochs=epochs, lr=lr)
                        
                        # Evaluate model on validation set
                        y_pred = model.predict(X_val)
                        acc = self.accuracy(y_val, y_pred)
                        
                        # Update best model if current model is better
                        if acc > best_acc and acc > 92:
                            best_acc = acc
                            best_model = model
                            print(f"New best model: hidden_layers={h}, hidden_size={hs}, lr={lr:.6f}, epochs={epochs}, accuracy={best_acc:.4f}")
                            return best_model
        
        
