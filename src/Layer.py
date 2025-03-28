import numpy as np
from Activation import Activation
class Layer:
    def __init__(self, input_size, output_size, activation, weight_init, **kwargs):
        """
        Inisialisasi layer neural network.
        
        Parameters:
        input_size (int): Jumlah neuron dari layer sebelumnya
        output_size (int): Jumlah neuron di layer ini
        activation (str): Fungsi aktivasi ('relu', 'sigmoid', 'softmax', dll)
        weight_init (dict): Konfigurasi inisialisasi bobot
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.kwargs = kwargs
        
        # Inisialisasi matriks bobot W dengan bentuk (input_size, output_size)
        self.W = self._initialize_weights()
        
        # Inisialisasi bobot bias untuk setiap neuron di layer saat ini
        self.b = np.zeros(output_size)
        
        # Gradien bobot dan bias
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Menyimpan nilai selama forward pass
        self.A_prev = None 
        self.Z = None        
        self.A = None        

    def _initialize_weights(self):
        if self.weight_init == 'zero':
            return np.zeros((self.input_size, self.output_size))
            
        elif self.weight_init == 'uniform':
            np.random.seed(self.kwargs.get('seed', None))
            return np.random.uniform(
                low=self.kwargs['lower'],
                high=self.kwargs['upper'],
                size=(self.input_size, self.output_size)
            )
            
        elif self.weight_init == 'normal':
            np.random.seed(self.kwargs.get('seed', None))
            return np.random.normal(
                loc=self.kwargs['mean'],
                scale=np.sqrt(self.kwargs['variance']),
                size=(self.input_size, self.output_size)
            )
        else:
            raise ValueError("Metode inisialisasi tidak valid")

    def forward(self, A_prev):
        """
        Forward pass
        A_prev: Aktivasi dari layer sebelumnya (bentuk: [batch_size, input_size])
        """
        self.A_prev = A_prev
        
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = np.maximum(0, self.Z)
        elif self.activation == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))
        elif self.activation == 'softmax':
            exp_Z = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))
            self.A = exp_Z / exp_Z.sum(axis=1, keepdims=True)
        else:
            self.A = self.Z 
        
        return self.A

    def backward(self, dA, learning_rate, m_batch):
        """
        Backprop 

        Parameter:
        dA (np.array): Gradien dari loss terhadap output layer ini
        learning_rate (float): Tingkat pembelajaran
        m_batch (int): Jumlah sampel dalam batch
        """
        # Hitung gradien terhadap Z
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0) 
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.Z))
            dZ = dA * sig * (1 - sig)
        elif self.activation == 'softmax':
            dZ = dA 
        else:
            dZ = dA  

        # Hitung gradien bobot dan bias
        self.dW = (self.A_prev.T @ dZ) / m_batch
        self.db = np.sum(dZ, axis=0) / m_batch

        # Hitung gradien untuk layer sebelumnya
        dA_prev = dZ @ self.W.T
        
        # Update bobot
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        return dA_prev

    def __repr__(self):
        return (f"Layer(input_size={self.input_size}, output_size={self.output_size}, "
                f"activation='{self.activation}', weight_init='{self.weight_init}', "
                f"W.shape={self.W.shape}, b.shape={self.b.shape})"
                f"W={self.W}"
                f"B={self.b}")