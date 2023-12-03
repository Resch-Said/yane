import bisect

from src.neural_network import YaneConfig, Connection
from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.NodeTypes import NodeTypes
from src.neural_network.exceptions.InvalidConnection import InvalidConnection

yane_config = YaneConfig.load_json_config()


class Node:
    ID = 0

    def __init__(self, node_type: NodeTypes, ID=None):
        self.value = 0.0
        self.next_connections = []
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))
        self.type = node_type
        self.id = ID

        if ID is None:
            self.id = Node.ID
            Node.ID += 1

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

    def get_next_connections(self) -> list[Connection]:
        return self.next_connections

    def add_next_connection(self, connection: Connection):
        if connection in self.next_connections:
            raise InvalidConnection("Cannot add connection twice")

        if connection.get_in_node() != self:
            raise InvalidConnection("Cannot add connection with different in neuron than this neuron")

        if connection.get_out_node() is None:
            raise InvalidConnection("Cannot add connection with no out neuron")

        for next_connection in self.next_connections:
            if next_connection.get_out_node() == connection.get_out_node():
                raise InvalidConnection("Cannot add connection with same out neuron twice")

        bisect.insort(self.next_connections, connection, key=lambda x: x.get_id())

    def activate(self):
        self.value = ActivationFunction.activate(self.activation, self.value)

    def reset(self):
        self.value = 0.0

    def get_id(self):
        return self.id

    # Avoid deep copy because of recursion
    def copy(self):
        new_node = Node(self.type, self.id)
        new_node.set_activation(self.activation)

        return new_node

    def fire(self):
        if self.type != NodeTypes.INPUT:
            self.activate()

        for connection in self.next_connections:
            next_node: Node = connection.get_out_node()
            next_node.set_value(next_node.get_value() + self.value * connection.get_weight())

        if self.type == NodeTypes.HIDDEN:
            self.value = 0.0

    def mutate_activation_function(self):
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))

    def remove_next_connection(self, con):
        self.next_connections.remove(con)
