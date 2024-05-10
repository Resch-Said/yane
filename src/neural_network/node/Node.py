from src.neural_network.connection import Connection, ConnectionCollection
from src.neural_network.node.NodeTypes import NodeTypes
from src.neural_network.util import YaneConfig
from src.neural_network.util.ActivationFunction import ActivationFunction

yane_config = YaneConfig.load_json_config()


class Node:
    global_node_id = 0
    global_input_pos = 0

    def __init__(self, node_type: NodeTypes, node_id=None, input_pos=None):
        self.value = 0.0
        from src.neural_network.connection.ConnectionCollection import ConnectionCollection

        # Target nodes are always the next nodes and source nodes are always the current nodes
        # This should avoid confusions. So next_connections and previous_connections have always the same source node
        self.next_connections: ConnectionCollection = ConnectionCollection()
        self.previous_connections: ConnectionCollection = ConnectionCollection()
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))
        self.type = node_type
        self.id = node_id
        self.input_pos = input_pos
        self.original_input_data = None

        if node_id is None:
            self.id = Node.global_node_id
            Node.global_node_id += 1

        if node_type == NodeTypes.INPUT and input_pos is None:
            self.input_pos = Node.global_input_pos
            Node.global_input_pos += 1

    def __str__(self):
        return "Neuron: " + str(self.id) + " Value: " + str(self.value) + " Activation: " + str(
            self.activation) + " Type: " + str(self.type) + " Input pos: " + str(self.input_pos)

    def set_value(self, value):
        self.value = value

    def set_activation(self, activation):
        self.activation = activation

    def get_value(self):
        return self.value

    def get_activation(self):
        return self.activation

    def get_next_connections(self) -> ConnectionCollection:
        return self.next_connections

    def add_connection(self, connection: Connection):
        connection_forward = connection
        connection_backward = connection.copy().switch_nodes()

        self.next_connections.add(connection_forward)
        connection_forward.get_target_node().previous_connections.add(connection_backward)

    def remove_connection(self, connection: Connection):
        self.next_connections.remove_by_id(connection.get_id())
        self.previous_connections.remove_by_id(connection.get_id())

        connection.get_target_node().next_connections.remove_by_id(connection.get_id())
        connection.get_target_node().previous_connections.remove_by_id(connection.get_id())

    def get_previous_connections(self) -> ConnectionCollection:
        return self.previous_connections

    def add_next_connection(self, connection: Connection):
        self.next_connections.add(connection)

    def add_previous_connection(self, connection: Connection):
        self.previous_connections.add(connection)

    def remove_next_connection(self, connection: Connection):
        self.next_connections.remove(connection)

    def remove_previous_connection(self, connection: Connection):
        self.previous_connections.remove(connection)

    def activate(self):
        self.value = ActivationFunction.activate(self.activation, self.value)

    def reset(self):
        self.value = 0.0

    def get_id(self):
        return self.id

    def copy(self) -> 'Node':
        new_node = Node(self.type, self.id, self.input_pos)
        new_node.set_activation(self.activation)
        new_node.set_original_input_data(self.original_input_data)
        new_node.next_connections = self.next_connections.copy()
        new_node.previous_connections = self.previous_connections.copy()

        return new_node

    def fire(self, keep_input=False):

        if self.type != NodeTypes.INPUT:
            self.activate()

        for connection in self.next_connections.get_all_connections():
            connection.get_target_node().set_value(
                connection.get_target_node().get_value() + connection.get_weight() * self.value)

        if self.type == NodeTypes.INPUT and not keep_input:
            self.value = 0.0
        elif self.type == NodeTypes.INPUT and keep_input:
            self.value = self.original_input_data
        elif self.type == NodeTypes.HIDDEN:
            self.value = 0.0

    def mutate_activation_function(self):
        self.activation = ActivationFunction.get_function(YaneConfig.get_random_activation_function(yane_config))

    def get_input_position(self):
        return self.input_pos

    def set_original_input_data(self, value):
        self.original_input_data = value

    def get_type(self):
        return self.type
