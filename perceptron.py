import numpy as np

# Function to normalize the data
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).transpose()

np.random.seed(1)

# Return an interval (-1.0, 1.0)
synaptic_weights = 2 * np.random.random((3, 1)) - 1 

print("\nRandom starting synaptic weights:")
print(synaptic_weights)

training_iterations = 10000

for iteration in range(training_iterations):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.transpose(), adjustments)

print("\nSynaptic weights after training:")
print(synaptic_weights)

print("\nOutputs after training:")
print(outputs)
