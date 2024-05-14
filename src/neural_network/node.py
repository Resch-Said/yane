from enum import Enum
from neural_network.connection import Connection
from neural_network.mutation import Mutation
from neural_network.util.activation import ActivationType, ActivationFunction


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Node:
    def __init__(self, value=0, node_type=NodeType.HIDDEN) -> None:
        self.value = value
        self.bias = 0
        self.type = node_type
        self.input_index = 0
        self.activation_function: ActivationType = ActivationType.SIGMOID

        self.connections: list[Connection] = []
        self.mutation_bias = Mutation()
        self.mutation_index = Mutation()
        self.mutation_function = Mutation()

    def mutate(self) -> None:
        self.bias = self.mutation_bias.mutate_value(self.bias)
        self.input_index = round(
            self.mutation_index.mutate_value(self.input_index))

        self.activation_function = self.mutation_function.mutate_custom(
            self.activation_function, ActivationType)

        for connection in self.connections:
            connection.mutate()

        self.mutation_bias.mutate_rates()
        self.mutation_index.mutate_rates()
        self.mutation_function.mutate_rates()

    def connect(self, neuron) -> None:
        for connection in self.connections:
            if connection.neuron == neuron:
                return
        self.connections.append(Connection(neuron))

    def disconnect(self, neuron=None, connection=None) -> None:
        if neuron is None and connection is None:
            raise ValueError("Either neuron or connection must be given")

        if connection is not None:
            self.connections.remove(connection)
            return

        for connection in self.connections:
            if connection.neuron == neuron:
                self.connections.remove(connection)

    def copy(self):
        neuron = Node()  # Don't copy value because copying is used for creating new genomes
        neuron.bias = self.bias
        neuron.type = self.type
        neuron.input_index = self.input_index
        neuron.activation_function = self.activation_function
        neuron.connections = [connection.copy()
                              for connection in self.connections]
        neuron.mutation_bias = self.mutation_bias.copy()
        neuron.mutation_index = self.mutation_index.copy()
        neuron.mutation_function = self.mutation_function.copy()
        return neuron

    def fire(self) -> list['Node']:
        next_nodes = []

        self.value = ActivationFunction.activate(
            self.activation_function, self.value)

        for connection in self.connections:
            connection.neuron.value += connection.weight * self.value + self.bias
            next_nodes.append(connection.neuron)

        return next_nodes
