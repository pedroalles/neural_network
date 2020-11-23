import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1- x)

    def train(self,  training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            outputs = self.think(training_inputs)
            error = training_outputs - outputs
            adjustments = np.dot(training_inputs.transpose(), error * self.sigmoid_derivative(outputs))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return outputs

if __name__ == "__main__":

    nn = NeuralNetwork()
    print('\nRandom synaptic weights: ')
    print(nn.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).transpose()

    training_iterations = 100000

    nn.train(training_inputs, training_outputs, training_iterations)

    print("\nSynaptic weights after training:")
    print(nn.synaptic_weights)

    A = str(input("\nInput 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))


    print("\nNew situation: input data =  ", A, B, C)
    print("Output data: ")
    print(nn.think(np.array([A, B, C])))