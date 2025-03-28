from FFNN import FFNN
weight_inits = [
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42},
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42},
]
# Initialize model
model = FFNN([2, 3, 1], ['relu', 'sigmoid'], 'categorical_cross_entropy', weight_inits=weight_inits)

# Visualize network structure
model.plot_as_graph(verbose=True)

weight_inits = [
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42},
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42},
    {'method': 'normal', 'mean': 0, 'variance': 0.01, 'seed': 42}
]
# Initialize model
model = FFNN([784, 128, 64, 10], ['relu', 'sigmoid'], 'categorical_cross_entropy', weight_inits=weight_inits)

# Visualize network structure
model.plot_as_graph(verbose=False)

# # Plot weight distributions for connection layers 0 and 1
# model.plot_weight_distribution([0, 1])

# # Plot gradient distributions for connection layer 1
# model.plot_gradient_distribution([1])