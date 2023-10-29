from copy import deepcopy, copy

from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron
from src.neural_network.YaneConfig import *


def mutate_weight(connection: Connection, weight_shift: float):
    if random.random() < 0.5:
        connection.weight += weight_shift
    else:
        connection.weight -= weight_shift


def get_total_fire_rate(working_neurons):
    total_fire_rate = 0
    for neuron in working_neurons:
        total_fire_rate += neuron.fire_rate_variable
    return total_fire_rate


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

    # TODO: Make random mutations smarter
    def random_mutate_weight(self, weight_shift: float):
        random_connection = random.choice(self.connections)
        mutate_weight(random_connection, weight_shift)

    def train(self, min_fitness):

        nn_parent = self

        nn_parent.forward_propagation()
        nn_child = nn_parent.create_child()
        current_fitness = nn_parent.get_fitness()

        while current_fitness < min_fitness:
            nn_child.forward_propagation()
            new_fitness = nn_child.get_fitness()

            if new_fitness > current_fitness:
                nn_parent = nn_child
                current_fitness = new_fitness
                print("New fitness: " + str(current_fitness))

            nn_child = deepcopy(nn_parent)
            nn_child.mutate()
        self.copy(nn_parent)

    def get_fitness(self):
        fitness = 0
        for neuron in self.output_neurons:
            fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    def clear_hidden_neurons(self):
        for neuron in self.hidden_neurons:
            neuron.value = 0.0

    def clear_output_neurons(self):
        for neuron in self.output_neurons:
            neuron.value = 0.0

    def clear_neurons(self):
        self.reset_input_neurons()
        self.reset_fire_rate()

        if get_clear_on_new_input():
            self.clear_hidden_neurons()
            self.clear_output_neurons()

    def add_input_neuron(self, neuron: Neuron = InputNeuron()):
        if type(neuron) is not InputNeuron:
            raise TypeError("Neuron is not of type InputNeuron")

        self.input_neurons.append(neuron)

    def add_hidden_neuron(self, neuron: Neuron = HiddenNeuron()):
        if type(neuron) is not HiddenNeuron:
            raise TypeError("Neuron is not of type HiddenNeuron")

        self.hidden_neurons.append(neuron)

    def add_output_neuron(self, neuron: Neuron = OutputNeuron()):
        if type(neuron) is not OutputNeuron:
            raise TypeError("Neuron is not of type OutputNeuron")

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

    def get_connection_weight(self, neuron_from, neuron_to):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                return connection.weight

    def get_neuron_forward_order(self):
        working_neurons = copy(self.input_neurons)

        for neuron in working_neurons:
            for next_neuron in self.get_connected_neurons_forward(neuron):
                if next_neuron not in working_neurons:
                    working_neurons.append(next_neuron)

        return working_neurons

    def reset_input_neurons(self):
        for neuron in self.input_neurons:
            neuron.value = neuron.value_fixed

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

    def set_expected_output_values(self, param):
        for expected_value, neuron in zip(param, self.output_neurons):
            neuron.expected_value = expected_value

    def copy(self, nn_current):
        self.input_neurons = deepcopy(nn_current.input_neurons)
        self.hidden_neurons = deepcopy(nn_current.hidden_neurons)
        self.output_neurons = deepcopy(nn_current.output_neurons)
        self.connections = deepcopy(nn_current.connections)

    def reset_fire_rate(self):
        for neuron in self.input_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed
        for neuron in self.hidden_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed
        for neuron in self.output_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed

    def set_all_activation_functions(self, linear):
        for neuron in self.input_neurons:
            neuron.activation_function = linear
        for neuron in self.hidden_neurons:
            neuron.activation_function = linear
        for neuron in self.output_neurons:
            neuron.activation_function = linear

    def forward_propagation(self):  # One tick cycle
        self.clear_neurons()

        working_neurons = self.get_neuron_forward_order()
        total_fire_rate = get_total_fire_rate(working_neurons)

        while total_fire_rate > 0:
            for neuron in working_neurons:
                if neuron.fire_rate_variable <= 0:
                    continue
                else:
                    neuron.fire_rate_variable -= 1
                    total_fire_rate -= 1
                    ActivationFunction.activate(neuron)

                for connection in self.get_connections(neuron):
                    connection.neuron_to.value += neuron.value * connection.weight

    def get_connections(self, neuron):
        connections = []
        for connection in self.connections:
            if connection.neuron_from == neuron:
                connections.append(connection)
        return connections

    def create_child(self):
        nn_child = deepcopy(self)
        nn_child.mutate()
        return nn_child

    def mutate(self):
        self.random_mutate_weight(get_random_weight_shift())

    def print(self):
        print("Input neurons: " + str(len(self.input_neurons)))
        print("Hidden neurons: " + str(len(self.hidden_neurons)))
        print("Output neurons: " + str(len(self.output_neurons)))
        print("Connections: " + str(len(self.connections)))
        print("")

        for neuron in self.input_neurons:
            print("Input neuron: " + str(neuron.value))

        for neuron in self.hidden_neurons:
            print("Hidden neuron: " + str(neuron.value))

        for neuron in self.output_neurons:
            print("Output neuron: " + str(neuron.value))

        for connection in self.connections:
            print("Connection: " + str(connection.weight))
