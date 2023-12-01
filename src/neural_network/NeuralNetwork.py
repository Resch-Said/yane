import bisect
import random

from src.neural_network import YaneConfig
from src.neural_network.Connection import Connection
from src.neural_network.Node import Node
from src.neural_network.NodeTypes import NodeTypes
from src.neural_network.exceptions.InvalidNode import InvalidNode
from src.neural_network.exceptions.InvalidNodeTypeException import InvalidNodeTypeException

yane_config = YaneConfig.load_json_config()


class NeuralNetwork:
    def __init__(self):
        self.last_weight_shift_connection = None
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.forward_order_list = None

    def get_all_nodes(self) -> list[Node]:
        return [node for node in self.input_nodes + self.hidden_nodes + self.output_nodes]

    def add_connection(self, connection):
        if self.get_all_nodes().__contains__(connection.get_in_node()) is False:
            raise InvalidNode("node in is not in the neural network")

        if self.get_all_nodes().__contains__(connection.get_out_node()) is False:
            raise InvalidNode("node out is not in the neural network")

        connection.get_in_node().add_next_connection(connection)

    def add_input_node(self, node: Node):
        if node.type is not NodeTypes.INPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode")

        bisect.insort(self.input_nodes, node, key=lambda x: x.get_id())

    def add_hidden_node(self, node: Node):
        if node.type is not NodeTypes.HIDDEN:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add HiddenNode")

        bisect.insort(self.hidden_nodes, node, key=lambda x: x.get_id())

    def add_output_node(self, node: Node):
        if node.type is not NodeTypes.OUTPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add OutputNode")

        bisect.insort(self.output_nodes, node, key=lambda x: x.get_id())

    def add_node(self, node: Node):

        if self.get_all_nodes().__contains__(node):
            raise InvalidNode("node already exists in the neural network")

        if node.type is NodeTypes.INPUT:
            self.add_input_node(node)
        elif node.type is NodeTypes.HIDDEN:
            self.add_hidden_node(node)
        elif node.type is NodeTypes.OUTPUT:
            self.add_output_node(node)
        else:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode, HiddenNode or OutputNode")

    def get_input_nodes(self) -> list[Node]:
        return self.input_nodes

    def get_hidden_nodes(self) -> list[Node]:
        return self.hidden_nodes

    def get_output_nodes(self) -> list[Node]:
        return self.output_nodes

    def get_node_by_id(self, node_id) -> Node | None:
        node: Node

        for node in self.get_all_nodes():
            if node.get_id() == node_id:
                return node

        return None

    def get_all_connections(self) -> list[Connection]:
        connections = []
        for node in self.get_all_nodes():
            connections += node.get_next_connections()
        return connections

    def set_input_data(self, data):
        while len(data) > len(self.input_nodes):
            new_node = Node(NodeTypes.INPUT)
            self.add_input_node(new_node)

        for i, v in enumerate(data):
            self.input_nodes[i].set_value(v)
        for i in range(len(data), len(self.input_nodes)):
            self.input_nodes[i].set_value(0.0)

    def forward_propagation(self, data=None):
        self.clear_output()

        if data is not None:
            self.set_input_data(data)

        for node in self.get_forward_order_list():
            node.fire()

        return self.get_output_data()

    def clear_values(self):
        for node in self.hidden_nodes:
            node.set_value(0.0)

        for node in self.output_nodes:
            node.set_value(0.0)

    def get_forward_order_list(self) -> list[Node]:

        if self.forward_order_list is not None:
            return self.forward_order_list

        self.forward_order_list = []

        for node in self.get_input_nodes():
            if len(node.get_next_connections()) > 0:
                self.forward_order_list.append(node)

        node: Node

        for node in self.forward_order_list:
            for connection in node.get_next_connections():
                if connection.get_out_node() not in self.forward_order_list:
                    self.forward_order_list.append(connection.get_out_node())

        return self.forward_order_list

    def get_output_data(self) -> list:
        output_data = []

        for node in self.output_nodes:
            output_data.append(node.get_value())

        return output_data

    def calculate_net_cost(self):
        net_cost = len(self.get_all_connections())
        net_cost += len(self.get_all_nodes())
        return net_cost

    def remove_all_connections(self):
        for node in self.get_all_nodes():
            node.next_connections = []

    def get_random_node(self):
        nodes = self.get_all_nodes()

        if len(nodes) > 0:
            return random.choice(nodes)
        else:
            return None

    def print(self):
        print("Neural Network:")
        print("Input nodes:")
        for node in self.input_nodes:
            print(node)
        print("Hidden nodes:")
        for node in self.hidden_nodes:
            print(node)
        print("Output nodes:")
        for node in self.output_nodes:
            print(node)
        print("Connections:")
        for connection in self.get_all_connections():
            print(connection)
        print("End of Neural Network")

    def get_input_data(self):
        input_data = []

        for node in self.input_nodes:
            input_data.append(node.get_value())

        return input_data

    def clear_output(self):
        for node in self.output_nodes:
            node.set_value(0.0)

    def get_last_weight_shift_connection(self) -> Connection:
        return self.last_weight_shift_connection
