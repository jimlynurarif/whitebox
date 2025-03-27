import numpy as np
class Loss:
    @staticmethod
    def mse(Y_pred, Y_true):
        return np.mean((Y_pred - Y_true)**2)
    @staticmethod
    def mse_derivative(Y_pred, Y_true):
        return 2 * (Y_pred - Y_true) / Y_pred.size
    
    @staticmethod
    def binary_cross_entropy(Y_pred, Y_true):
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    @staticmethod
    def binary_cross_entropy_derivative(Y_pred, Y_true):
        return (Y_pred - Y_true) / (Y_pred * (1 - Y_pred)) / Y_true.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(Y_pred, Y_true):
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))
    @staticmethod
    def categorical_cross_entropy_derivative(Y_pred, Y_true):
        return (Y_pred - Y_true) / Y_true.shape[0]