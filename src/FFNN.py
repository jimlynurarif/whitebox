import matplotlib.pyplot as plt
from graphviz import Digraph
import networkx as nx
from IPython.display import Image, display  # For Jupyter notebooks
import tempfile
import sys, os
import pickle
import random
from pathlib import Path
import string
import numpy as np
from Activation import Activation
from Layer import Layer
from Loss import Loss

class FFNN:
    def __init__(self, layer_sizes, activations, loss_function, weight_inits):
        """
        Inisialisasi model FFNN
        
        Parameters:
        layer_sizes (list): Daftar ukuran tiap layer 
        activations (list): Daftar fungsi aktivasi tiap layer
        loss_function (str): Fungsi loss yang digunakan
        weight_inits (list): Daftar konfigurasi inisialisasi bobot
        """
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
                **kwargs  
            )
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
            return 1 
    
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
    def plot_as_graph(self, verbose=False):
        """
        Visualizes the neural network as a graph using Graphviz Digraph.
        Shows neurons as nodes and connections as edges with weights/gradients.
        """
        dot = Digraph(comment='Neural Network')
        dot.attr(rankdir='LR', splines='line')  # Left-right layout with straight lines
        
        if verbose:
            # Create nodes for each neuron
            for layer_idx, size in enumerate(self.layer_sizes):
                with dot.subgraph(name=f'cluster_{layer_idx}') as sg:
                    sg.attr(rank='same', style='invis')  # Invisible cluster for alignment
                    for neuron_idx in range(size):
                        if layer_idx == 0:
                            sg.node(f'L{layer_idx}_N{neuron_idx}', 
                                label=f'Input {neuron_idx}')
                        elif layer_idx == len(self.layer_sizes) - 1:
                            sg.node(f'L{layer_idx}_N{neuron_idx}', 
                                label=f'Output {neuron_idx}')
                        else:
                            sg.node(f'L{layer_idx}_N{neuron_idx}', 
                                label=f'Layer {layer_idx}\nNeuron {neuron_idx}')
            # Create edges with weights and gradients
            for conn_idx, layer in enumerate(self.layers):
                input_size = layer.input_size
                output_size = layer.output_size
                
                for i in range(input_size):
                    for j in range(output_size):
                        src = f'L{conn_idx}_N{i}'
                        dest = f'L{conn_idx+1}_N{j}'
                        
                        # Get weight and gradient values
                        weight = layer.W[i, j]
                        grad = layer.dW[i, j] if hasattr(layer, 'dW') and layer.dW is not None else 0.0
                        
                        dot.edge(src, dest, 
                                label=f'W: {weight:.2f}\nG: {grad:.2f}', 
                                fontsize='8', 
                                arrowsize='0.5')
        else:
            # Simplified nodes for layers
            for i, size in enumerate(self.layer_sizes):
                label = (
                    f"Layer {i}\n"
                    f"Size: {size}\n"
                    f"{self.activations[i] if i<len(self.activations) else ''}"
                )
                dot.node(f'L{i}', label=label, shape='box', style='filled', fillcolor='#f0f0f0')
            
            # Simplified connections between layers
            for i in range(len(self.layer_sizes)-1):
                with dot.subgraph() as s:
                    s.attr(label=f"W: {self.layers[i].W.shape} | dW: {self.layers[i].dW.shape if hasattr(self.layers[i], 'dW') else 'N/A'}")
                    s.edge(f'L{i}', f'L{i+1}')
    
        # Display the graph
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Generate random filename
                rand_name = ''.join(random.choices(string.ascii_letters, k=10))
                filepath = Path(tmpdir) / rand_name
                
                # Render to temporary directory
                dot.render(filename=str(filepath), 
                        format='png',
                        cleanup=True,
                        view=False)
                
                # Read and display the image
                img_path = filepath.with_suffix('.png')
                
                if 'IPython' in sys.modules:
                    display(Image(str(img_path)))
                else:
                    img = plt.imread(img_path)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                    
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            print("Attempting direct rendering...")
            dot.render('network_graph_fallback', view=True, format='png', cleanup=True)
    def plot_weight_distribution(self, layers):
        """Plots weight distribution for specified connection layers"""
        self.plot_distribution(layers, is_weight=True)

    def plot_gradient_distribution(self, layers):
        """Plots gradient distribution for specified connection layers"""
        self.plot_distribution(layers, is_weight=False)
        
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