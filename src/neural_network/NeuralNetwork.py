import random
from copy import deepcopy, copy

from src.neural_network.Connection import Connection
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron


def mutate_weight(connection: Connection, weight_shift: float):
    if random.random() < 0.5:
        connection.weight += weight_shift
    else:
        connection.weight -= weight_shift


class NeuralNetwork:
    def __init__(self):
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        self.connections = []

    def get_connection_between_neurons(self, neuron_from: Neuron, neuron_to: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                return connection
        return None

    def random_mutate_weight(self, weight_shift: float):
        random_connection = random.choice(self.connections)
        mutate_weight(random_connection, weight_shift)

    # TODO: Implement
    # Make copy of network
    # Test Fitness
    # Mutate
    # Check if better
    # If better, replace old network with new one
    # Repeat
    def train(self, number_iterations: int, weight_shift: float):

        nn_current = deepcopy(self)
        nn_current.forward_propagation()
        nn_new = deepcopy(nn_current)

        for i in range(number_iterations):
            nn_new.random_mutate_weight(weight_shift)
            nn_new.forward_propagation()

            if nn_current.get_fitness() < nn_new.get_fitness():
                nn_current = deepcopy(nn_new)
                print("Better network found")
            else:
                nn_new = deepcopy(nn_current)

        self.copy(nn_current)

    def get_fitness(self):
        fitness = 0
        for neuron in self.output_neurons:
            fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    def reset_neurons(self):
        for neuron in self.hidden_neurons:
            neuron.value = 0.0
        for neuron in self.output_neurons:
            neuron.value = 0.0

    def add_input_neuron(self, neuron: Neuron = InputNeuron()):
        self.input_neurons.append(neuron)

    def add_hidden_neuron(self, neuron: Neuron):
        self.hidden_neurons.append(neuron)

    def add_output_neuron(self, neuron: Neuron):
        self.output_neurons.append(neuron)

    def remove_neuron(self, neuron: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron or connection.neuron_to == neuron:
                self.connections.remove(connection)

        if neuron in self.input_neurons:
            self.input_neurons.remove(neuron)
        elif neuron in self.hidden_neurons:
            self.hidden_neurons.remove(neuron)
        elif neuron in self.output_neurons:
            self.output_neurons.remove(neuron)

    def add_connection(self, neuron_from: Neuron, neuron_to: Neuron, weight=1.0):
        connection = Connection(neuron_from, neuron_to, weight)
        self.connections.append(connection)

    def get_connected_neurons_forward(self, neuron: Neuron):
        connected_neurons = []
        for connection in self.connections:
            if connection.neuron_from == neuron:
                connected_neurons.append(connection.neuron_to)
        return connected_neurons

    def get_connected_neurons_backward(self, neuron: Neuron):
        connected_neurons = []
        for connection in self.connections:
            if connection.neuron_to == neuron:
                connected_neurons.append(connection.neuron_from)
        return connected_neurons

    def remove_connection(self, neuron_from: Neuron, neuron_to: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                self.connections.remove(connection)

    def forward_propagation(self):
        self.reset_neurons()

        open_list = self.output_neurons.copy()
        closed_list = []
        active_neuron_can_be_closed = False

        while len(open_list) > 0:
            for active_neuron in open_list:
                active_neuron_can_be_closed = True
                for neuron_backwards in self.get_connected_neurons_backward(active_neuron):
                    if neuron_backwards not in closed_list:
                        active_neuron_can_be_closed = False
                        if neuron_backwards not in open_list:
                            open_list.append(neuron_backwards)
                if active_neuron_can_be_closed:
                    open_list.remove(active_neuron)
                    closed_list.append(active_neuron)
                    for neuron_backwards in self.get_connected_neurons_backward(active_neuron):
                        active_neuron.value += neuron_backwards.value * self.get_connection_weight(neuron_backwards,
                                                                                                   active_neuron)

        return self

    def get_output_values(self):
        output_values = []
        for neuron in self.output_neurons:
            output_values.append(neuron.value)
        return output_values

    def get_expected_output_values(self):
        expected_output_values = []
        for neuron in self.output_neurons:
            expected_output_values.append(neuron.expected_value)
        return expected_output_values

    def get_connection_weight(self, neuron_backwards, active_neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron_backwards and connection.neuron_to == active_neuron:
                return connection.weight

    def set_expected_output_values(self, param):
        for expected_value, neuron in zip(param, self.output_neurons):
            neuron.expected_value = expected_value

    def copy(self, nn_current):
        self.input_neurons = copy(nn_current.input_neurons)
        self.hidden_neurons = copy(nn_current.hidden_neurons)
        self.output_neurons = copy(nn_current.output_neurons)
        self.connections = copy(nn_current.connections)
