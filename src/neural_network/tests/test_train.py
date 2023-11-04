import numpy as np

from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.NeuroEvolution import NeuroEvolution


def test_train():
    yane = NeuroEvolution(output_neurons=1)

    def custom_evaluation(self):
        fitness = 0
        input_data = [2, 4]
        expected_output = [6]
        output_data = self.forward_propagation(input_data)

        for i, v in enumerate(output_data):
            fitness -= np.abs(expected_output[i] - v)

        return fitness

    NeuralNetwork.custom_evaluation = custom_evaluation

    yane.train(min_fitness=-0.01)
