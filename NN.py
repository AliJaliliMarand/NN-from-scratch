import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers=[], hidden_size=4):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        activations = []
        for i in range(len(self.weights)):
            if i == 0:
                Z = np.dot(X, self.weights[i]) + self.biases[i]
            else:
                Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                activation = self.softmax(Z)
            else:
                activation = self.relu(Z)
            activations.append(activation)
        return activations
    
    def backprop(self, X, y, activations):
        dweights = []
        dbiases = []
        error = activations[-1] - y
        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights) - 1:
                dW = np.dot(activations[i-1].T, error)
            else:
                dZ = (activations[i] > 0).astype(int) * np.dot(error, self.weights[i+1].T)
                if i == 0:
                    dW = np.dot(X.T, dZ)
                else:
                    dW = np.dot(activations[i-1].T, dZ)
            db = np.sum(error, axis=0, keepdims=True)
            dweights.insert(0, dW)
            dbiases.insert(0, db)
            error = dZ
        return dweights, dbiases
    
    def train(self, X, y, epochs=100, lr=0.1):
        for epoch in range(epochs):
            activations = self.forward(X)
            dweights, dbiases = self.backprop(X, y, activations)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * dweights[i]
                self.biases[i] -= lr * dbiases[i]
    
   

    def tuner(self, X, y, hidden_layers=[], hidden_size_range=(2, 10), epochs_range=(10, 100), lr_range=(0.001, 1)):
        best_acc = 0
        best_model = None
        
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split data into training and validation sets
        split = 0.8  # 80% training, 20% validation
        split_idx = int(len(X) * split)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        for h in hidden_layers:
            for hs in range(hidden_size_range[0], hidden_size_range[1]+1):
                for lr in np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num=5):
                    for epochs in range(epochs_range[0], epochs_range[1]+1):
                        # Train model with current hyperparameters
                        model = NeuralNetwork(X.shape[1], y.shape[1], hidden_layers=[hs]*h, lr=lr)
                        model.train(X_train, y_train, epochs=epochs)
                        
                        # Evaluate model on validation set
                        y_pred = model.predict(X_val)
                        acc = self.accuracy(y_val, y_pred)
                        
                        # Update best model if current model is better
                        if acc > best_acc:
                            best_acc = acc
                            best_model = model
                            print(f"New best model: hidden_layers={h}, hidden_size={hs}, lr={lr:.6f}, epochs={epochs}, accuracy={best_acc:.4f}")
        
        return best_model

