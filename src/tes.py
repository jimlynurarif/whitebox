from FFNN import FFNN
import numpy as np
from Layer import Layer
# layer_sizes = [5, 6, 1]
# activations = ['relu', 'softmax']
# weight_inits = [
#     {'method': 'zero', 'mean': 0, 'variance': 0.01, 'seed': 42},
#     {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42}
# ]
# loss_function = 'categorical_cross_entropy'

# model = FFNN(
#     layer_sizes=layer_sizes,
#     activations=activations,
#     loss_function=loss_function,
#     weight_inits=weight_inits
# )

layer = Layer(
    input_size=2,
    output_size=2,
    activation='sigmoid',
    weight_init='zero',
    method='zero'
)
layer.W = np.array([[0.4, 0.5], [0.45, 0.55]])  # Manual set bobot
layer.b = np.array([0.6, 0.6])  # Manual set bias

# Input dummy
X = np.array([[0.5933, 0.5969]])  # Bentuk: (1, 2)

# Forward pass
output = layer.forward(X)
print(output)