import numpy as np
class Activation:
    @staticmethod
    def linear(Z):
        return Z
    @staticmethod
    def linear_derivative(Z):
        return np.ones_like(Z)
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)
    
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    @staticmethod
    def sigmoid_derivative(Z):
        sig = Activation.sigmoid(Z)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(Z):
        return np.tanh(Z)
    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.tanh(Z)**2
    
    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)
    @staticmethod
    def softmax_derivative(Z):
        raise NotImplementedError("Gunakan kombinasi softmax dan cross-entropy untuk efisiensi")