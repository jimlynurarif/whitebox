import matplotlib.pyplot as plt
import pickle
import numpy as np
from Activation import Activation
from Layer import Layer
from Loss import Loss

class FFNN:
    def __init__(self, layer_sizes, activations, loss_function, weight_inits):
        
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.layers = []
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            activation = activations[i]
            
            weight_init_config = weight_inits[i]
            method = weight_init_config['method']
            kwargs = {k: v for k, v in weight_init_config.items() if k != 'method'}
            
            layer = Layer(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
                weight_init=method,
                **kwargs  # Parameter tambahan
            )
            # print(layer)
            self.layers.append(layer)
    
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, X, Y, learning_rate):
        m = Y.shape[0]
        
        # Hitung gradien loss terhadap output
        if self.loss_function == 'categorical_cross_entropy' and self.layers[-1].activation == 'softmax':
            # Special case: Softmax + CrossEntropy
            dZ = self.layers[-1].A - Y
        else:
            # Hitung turunan loss biasa
            if self.loss_function == 'mse':
                dA = Loss.mse_derivative(self.layers[-1].A, Y)
            elif self.loss_function == 'categorical_cross_entropy':
                dA = Loss.categorical_cross_entropy_derivative(self.layers[-1].A, Y)
            
            # Hitung dZ untuk output layer
            dZ = dA * self._activation_derivative(self.layers[-1].activation, self.layers[-1].Z)
        
        # Backpropagate melalui semua layer
        for i in reversed(range(len(self.layers)-1)):  # Mulai dari layer sebelum output
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            
            # Hitung dA untuk layer saat ini
            dA = dZ @ next_layer.W.T
            
            # Hitung dZ untuk layer saat ini
            dZ = dA * self._activation_derivative(layer.activation, layer.Z)
            
            # Update gradien
            layer.dW = (layer.A_prev.T @ dZ) / m
            layer.db = np.sum(dZ, axis=0) / m
            
            # Update bobot
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
    
    def _activation_derivative(self, activation, Z):
        if activation == 'linear':
            return Activation.linear_derivative(Z)
        elif activation == 'relu':
            return Activation.relu_derivative(Z)
        elif activation == 'sigmoid':
            return Activation.sigmoid_derivative(Z)
        elif activation == 'tanh':
            return Activation.tanh_derivative(Z)
        elif activation == 'softmax':
            return 1  # Diasumsikan sudah dihitung di loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.01, verbose=0):
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            # Mini-batch
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate)
            
            # Hitung loss
            train_pred = self.forward(X_train)
            train_loss = self._compute_loss(train_pred, y_train)
            val_pred = self.forward(X_val)
            val_loss = self._compute_loss(val_pred, y_val)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return history
    
    def _compute_loss(self, Y_pred, Y_true):
        if self.loss_function == 'mse':
            return Loss.mse(Y_pred, Y_true)
        elif self.loss_function == 'binary_cross_entropy':
            return Loss.binary_cross_entropy(Y_pred, Y_true)
        elif self.loss_function == 'categorical_cross_entropy':
            return Loss.categorical_cross_entropy(Y_pred, Y_true)
    
    def plot_distribution(self, layers, is_weight=True):
        for layer_idx in layers:
            layer = self.layers[layer_idx]
            data = layer.W.flatten() if is_weight else layer.dW.flatten()
            plt.hist(data, bins=50)
            plt.title(f"Layer {layer_idx} {'Weight' if is_weight else 'Gradient'} Distribution")
            plt.show()
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)