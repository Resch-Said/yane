from copy import deepcopy

from src.neural_network.ActivationFunction import ActivationFunction


class Neuron:
    ID = 0

    def __init__(self):
        self.value = 0.0
        self.bias = 1.0
        self.next_connections = []
        self.activation = ActivationFunction.SIGMOID
        self.id = Neuron.ID
        Neuron.ID += 1

    def __str__(self):
        return "Neuron: " + str(self.id) + " Value: " + str(self.value) + " Bias: " + str(
            self.bias) + " Activation: " + str(self.activation)

    def set_value(self, value):
        self.value = value

    def set_bias(self, bias):
        self.bias = bias

    def set_activation(self, activation):
        self.activation = activation

    def get_value(self):
        return self.value

    def get_bias(self):
        return self.bias

    def get_activation(self):
        return self.activation

    def get_next_connections(self) -> list:
        return self.next_connections

    def add_next_connection(self, connection):
        self.next_connections.append(connection)

    def activate(self):
        self.value = ActivationFunction.activate(self.activation, self.value)

    def reset(self):
        self.value = 0.0

    def get_id(self):
        return self.id

    def copy(self):
        return deepcopy(self)

    def fire(self):
        for connection in self.next_connections:
            next_neuron: Neuron = connection.get_out_neuron()
            next_neuron.set_value(next_neuron.get_value() + self.value * connection.get_weight())

    def set_connections(self, connections):
        self.next_connections = connections
