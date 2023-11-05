from copy import deepcopy

from src.neural_network import YaneConfig
from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.exceptions.InvalidConnection import InvalidConnection

yane_config = YaneConfig.load_json_config()


class Neuron:
    ID = 0

    def __init__(self):
        self.value = 0.0
        self.next_connections = []
        self.activation = ActivationFunction.SIGMOID
        self.id = Neuron.ID
        Neuron.ID += 1

    def __str__(self):
        return "Neuron: " + str(self.id) + " Value: " + str(self.value) + " Activation: " + str(self.activation)

    def set_value(self, value):
        self.value = value

    def set_activation(self, activation):
        self.activation = activation

    def get_value(self):
        return self.value

    def get_activation(self):
        return self.activation

    def get_next_connections(self) -> list:
        return self.next_connections

    def add_next_connection(self, connection):
        if connection in self.next_connections:
            raise InvalidConnection("Cannot add connection twice")

        if connection.get_in_neuron() != self:
            raise InvalidConnection("Cannot add connection with different in neuron than this neuron")

        if connection.get_out_neuron() is None:
            raise InvalidConnection("Cannot add connection with no out neuron")

        for next_connection in self.next_connections:
            if next_connection.get_out_neuron() == connection.get_out_neuron():
                raise InvalidConnection("Cannot add connection with same out neuron twice")

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
        self.activate()

        for connection in self.next_connections:
            next_neuron: Neuron = connection.get_out_neuron()
            next_neuron.set_value(next_neuron.get_value() + self.value * connection.get_weight())

    def mutate_activation_function(self):
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))

    def remove_next_connection(self, con):
        self.next_connections.remove(con)
