from FFNN import FFNN
layer_sizes = [5, 6, 1]
activations = ['relu', 'softmax']
weight_inits = [
    {'method': 'zero', 'mean': 0, 'variance': 0.01, 'seed': 42},
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42}
]
loss_function = 'categorical_cross_entropy'

model = FFNN(
    layer_sizes=layer_sizes,
    activations=activations,
    loss_function=loss_function,
    weight_inits=weight_inits
)