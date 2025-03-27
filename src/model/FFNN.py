import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Activation Functions
class Activation:
    @staticmethod
    def linear(x, derivative=False):
        return np.ones_like(x) if derivative else x

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig

    @staticmethod
    def tanh(x, derivative=False):
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2 if derivative else tanh_x
    
    @staticmethod
    def softmax(x, derivative=False):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_vals = exps / np.sum(exps, axis=1, keepdims=True)
        if derivative:
            return softmax_vals * (1 - softmax_vals)
        return softmax_vals

# Loss Functions
class Loss:
    @staticmethod
    def mse(y_true, y_pred, derivative=False):
        if derivative:
            return (y_pred - y_true) / y_true.shape[0]
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, derivative=False):
        eps = 1e-10  # Avoid log(0)
        if derivative:
            return (y_pred - y_true) / (y_pred * (1 - y_pred) + eps)
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, derivative=False):
        eps = 1e-10
        if derivative:
            return -y_true / (y_pred + eps)
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# Weight Initialization
class Initializer:
    @staticmethod
    def zeros(shape):
        return np.zeros(shape)
    
    @staticmethod
    def uniform(shape, lower=-0.1, upper=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(lower, upper, shape)
    
    @staticmethod
    def normal(shape, mean=0.0, variance=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, np.sqrt(variance), shape)

# Feedforward Neural Network
class FFNN:
    def __init__(self, layers, activations, loss, init_method='normal', init_params={}):
        self.layers = layers
        self.activations = [getattr(Activation, act) for act in activations]
        self.loss_function = getattr(Loss, loss)
        
        # Initialize Weights
        self.weights = []
        self.biases = []
        self.gradients = []
        
        for i in range(len(layers) - 1):
            shape = (layers[i], layers[i + 1])
            init_func = getattr(Initializer, init_method)
            self.weights.append(init_func(shape, **init_params))
            self.biases.append(np.zeros((1, layers[i + 1])))
            self.gradients.append(np.zeros_like(self.weights[-1]))

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.activations[i](z))
        return self.a[-1]
    
    def backward(self, X, y):
        batch_size = X.shape[0]
        y_pred = self.forward(X)
        
        delta = self.loss_function(y, y_pred, derivative=True) * self.activations[-1](y_pred, derivative=True)
        self.gradients[-1] = np.dot(self.a[-2].T, delta) / batch_size
        
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activations[i](self.a[i + 1], derivative=True)
            self.gradients[i] = np.dot(self.a[i].T, delta) / batch_size
    
    def update_weights(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.gradients[i]
    
    def train(self, X_train, y_train, batch_size, epochs, lr, verbose=1):
        history = {'loss': []}
        num_batches = X_train.shape[0] // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[indices], y_train[indices]
            
            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]
                self.backward(X_batch, y_batch)
                self.update_weights(lr)
                epoch_loss += self.loss_function(y_batch, self.forward(X_batch))
            
            history['loss'].append(epoch_loss / num_batches)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
