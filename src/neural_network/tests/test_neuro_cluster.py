from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.NeuroCluster import NeuroCluster
from src.neural_network.TrainingData import get_data_size, get_input_data, get_output_data


def test_train_dataset():
    nc = NeuroCluster()

    def custom_fitness(self):
        fitness = 0

        for j in range(get_data_size()):
            self.set_input_neurons(get_input_data(j))
            self.set_expected_output_values(get_output_data(j))
            self.forward_propagation()

            for neuron in self.output_neurons:
                fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    NeuralNetwork.custom_fitness = custom_fitness
    
    nc.train_population()
    # nc.print()

    assert nc.neural_networks[0].fitness > -0.1


def test_train_simple():
    nc = NeuroCluster()

    nc.set_input_neurons([5, 10])
    nc.set_expected_output_values(list(range(100)))

    nc.train_population()

    nc.neural_networks[0].print()

    assert nc.neural_networks[0].fitness > -0.1


def test_train_simple_2():
    nc = NeuroCluster()

    nc.set_input_neurons([5, 10])
    nc.set_expected_output_values(list(range(20)))

    nc.train_population()

    nc.neural_networks[0].print()

    assert nc.neural_networks[0].fitness > -0.1


def test_train_simple_3():
    nc = NeuroCluster()

    nc.set_input_neurons([5, 10])
    nc.set_expected_output_values(list(range(5)))

    nc.train_population()

    nc.neural_networks[0].print()

    assert nc.neural_networks[0].fitness > -0.1
