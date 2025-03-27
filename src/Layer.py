import numpy as np
from Activation import Activation
class Layer:
    def __init__(self, input_size, output_size, activation, weight_init, **kwargs):
        """
        input_size: Jumlah neuron di layer sebelumnya
        output_size: Jumlah neuron di layer saat ini
        W[i][j]: Bobot dari neuron ke-i (layer sebelumnya) ke neuron ke-j (layer saat ini)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.kwargs = kwargs
        
        # Inisialisasi matriks bobot W dengan bentuk (input_size, output_size)
        self.W = self._initialize_weights()
        
        # Inisialisasi bias untuk setiap neuron di layer saat ini
        self.b = np.zeros(output_size)
        
        # Gradien bobot dan bias
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Menyimpan nilai selama forward pass
        self.A_prev = None  # Aktivasi layer sebelumnya
        self.Z = None        # Input sebelum aktivasi
        self.A = None        # Output setelah aktivasi

    def _initialize_weights(self):
        """Inisialisasi matriks W dengan orientasi [source][target]"""
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
        
        # Hitung input ke fungsi aktivasi: Z = A_prev . W + b
        # A_prev: [batch_size, input_size]
        # W:      [input_size, output_size]
        # Z:      [batch_size, output_size]
        self.Z = np.dot(A_prev, self.W) + self.b
        # Terapkan fungsi aktivasi
        if self.activation == 'relu':
            self.A = np.maximum(0, self.Z)
        elif self.activation == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))
        elif self.activation == 'softmax':
            exp_Z = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))
            self.A = exp_Z / exp_Z.sum(axis=1, keepdims=True)
        else:
            self.A = self.Z  # Linear
        
        return self.A

    def backward(self, dA, learning_rate, m_batch):
        """
        dA: Gradien dari LOSS terhadap output layer ini (A)
        m_batch: Jumlah sampel dalam batch
        """
        # Hitung gradien terhadap Z
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)  # Pastikan dA dan Z memiliki dimensi yang sama
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.Z))
            dZ = dA * sig * (1 - sig)
        elif self.activation == 'softmax':
            dZ = dA  # Sudah dihitung di loss function
        else:
            dZ = dA  # Linear

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