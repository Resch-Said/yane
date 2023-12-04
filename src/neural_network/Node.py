import bisect

from src.neural_network import YaneConfig, Connection
from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.NodeTypes import NodeTypes
from src.neural_network.exceptions.InvalidConnection import InvalidConnection

yane_config = YaneConfig.load_json_config()


class Node:
    ID = 0
    INPUT_POS = 0

    def __init__(self, node_type: NodeTypes, ID=None, input_pos=None):
        self.value = 0.0
        self.next_connections = []
        self.previous_connections = []
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))
        self.type = node_type
        self.id = ID
        self.input_pos = input_pos
        self.original_input_data = None

        if ID is None:
            self.id = Node.ID
            Node.ID += 1

        if node_type == NodeTypes.INPUT and input_pos is None:
            self.input_pos = Node.INPUT_POS
            Node.INPUT_POS += 1

    def __str__(self):
        return "Neuron: " + str(self.id) + " Value: " + str(self.value) + " Activation: " + str(
            self.activation) + " Type: " + str(self.type) + " Input pos: " + str(self.input_pos)

    def set_value(self, value):
        self.value = value

    def set_activation(self, activation):
        self.activation = activation

    def get_next_nodes_and_weights(self):
        return [(connection.get_out_node(), connection.get_weight()) for connection in self.next_connections]

    def get_value(self):
        return self.value

    def get_activation(self):
        return self.activation

    def get_next_connections(self) -> list[Connection]:
        return self.next_connections

    def get_previous_connections(self) -> list[Connection]:
        return self.previous_connections

    def add_connection(self, connection: Connection):
        if connection.get_in_node() == self:
            self.add_next_connection(connection)
            connection.get_out_node().add_previous_connection(connection)
        elif connection.get_out_node() == self:
            self.add_previous_connection(connection)
            connection.get_in_node().add_next_connection(connection)
        else:
            raise InvalidConnection("Cannot add connection with different in or out neurons than this neuron")

    def add_previous_connection(self, connection: Connection):
        if connection in self.previous_connections:
            raise InvalidConnection("Cannot add connection twice")

        if connection.get_out_node() != self:
            raise InvalidConnection("Cannot add connection with different out neuron than this neuron")

        if connection.get_in_node() is None:
            raise InvalidConnection("Cannot add connection with no in neuron")

        for previous_connection in self.previous_connections:
            if previous_connection.get_in_node() == connection.get_in_node():
                raise InvalidConnection("Cannot add connection with same in neuron twice")

        bisect.insort(self.previous_connections, connection, key=lambda x: x.get_id())

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
        new_node = Node(self.type, self.id, self.input_pos)
        new_node.set_activation(self.activation)
        return new_node

    def fire(self, keep_input=False):
        if self.type != NodeTypes.INPUT:
            self.activate()

        next_nodes_and_weights = self.get_next_nodes_and_weights()
        [next_node.set_value(next_node.get_value() + self.value * weight) for next_node, weight in
         next_nodes_and_weights]

        if self.type == NodeTypes.INPUT and not keep_input:
            self.value = 0.0
        elif self.type == NodeTypes.INPUT and keep_input:
            self.value = self.original_input_data
        elif self.type == NodeTypes.HIDDEN:
            self.value = 0.0

    def mutate_activation_function(self):
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))

    def remove_next_connection(self, connection):
        self.next_connections.remove(connection)

    def remove_previous_connection(self, connection):
        self.previous_connections.remove(connection)

    def remove_connection(self, connection: Connection):
        if connection in self.next_connections:
            self.remove_next_connection(connection)
            connection.get_out_node().remove_previous_connection(connection)
        elif connection in self.previous_connections:
            self.remove_previous_connection(connection)
            connection.get_in_node().remove_next_connection(connection)

    def get_input_position(self):
        return self.input_pos

    def set_original_input_data(self, value):
        self.original_input_data = value
